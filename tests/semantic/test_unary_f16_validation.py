"""Validation and op_sel semantic tests for all 12 unary f16 VALU ops.

The op_sel results run on the CPU interpreter; the spirv-as / spirv-val checks of the same modules are separate
tests that skip without SPIRV-Tools on PATH (CI requires it through test_spirv_tools_required.py)."""
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402
import rdna2spirv  # noqa: E402

UNARY_F16_OPS = (
    "v_rsq_f16",
    "v_rcp_f16",
    "v_sqrt_f16",
    "v_exp_f16",
    "v_log_f16",
    "v_floor_f16",
    "v_fract_f16",
    "v_ceil_f16",
    "v_trunc_f16",
    "v_rndne_f16",
    "v_sin_f16",
    "v_cos_f16",
)
HAVE_SPIRV_TOOLS = bool(shutil.which("spirv-as") and shutil.which("spirv-val"))


def f2h(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def validate_spirv(spv_text):
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.2", asm_p, "-o", bin_p],
            capture_output=True,
            text=True,
        )
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", bin_p],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def run_ps(lines, name="ps_unary_f16_val"):
    return mrt0_words(translate_lines(name, lines))


# The op_sel programs, shared by the interpreter tests and the spirv-val test.
OPSEL_V4 = (f2h(0.25) << 16) | f2h(0.0)   # lo = 0.0 turns, hi = 0.25 turns
OPSEL_OLD = 0xBEEF1234
OPSEL_COMBOS = ((0, 0), (1, 0), (0, 1), (1, 1))


def opsel_combo_programs():
    """(op, suffix, name, lines): each op_sel:[shi,dhi] combination of v_sin/v_cos_f16 into v0..v3."""
    for op in ("v_sin_f16", "v_cos_f16"):
        for suffix in ("_e64", ""):
            lines = [f"v_mov_b32 v4, {OPSEL_V4:#x}"]
            for idx, (shi, dhi) in enumerate(OPSEL_COMBOS):
                lines.append(f"v_mov_b32 v{idx}, {OPSEL_OLD:#x}")
                lines.append(f"{op}{suffix} v{idx}, v4 op_sel:[{shi},{dhi}]")
            lines.append("exp mrt0 v0, v1, v2, v3 done vm")
            lines.append("s_endpgm")
            yield op, suffix, f"ps_{op}_{suffix or 'bare'}_opsel", lines


# lo = 0.125 turns (45 deg): sin ~= 0.707, cos ~= 0.707
# hi = 0.75  turns (270 deg): sin = -1.0,  cos = 0.0
MODS_OLD = 0xCAFE5678
MODS_V4 = (f2h(0.75) << 16) | f2h(0.125)
MODS_LINES = [
    f"v_mov_b32 v4, {MODS_V4:#x}",
    f"v_mov_b32 v0, {MODS_OLD:#x}",
    "v_sin_f16_e64 v0, v4 op_sel:[1,1]",            # hi = sin(0.75) = -1.0, lo preserved
    f"v_mov_b32 v1, {MODS_OLD:#x}",
    "v_sin_f16_e64 v1, -v4 op_sel:[1,0]",           # lo = sin(-0.75) = +1.0, hi preserved
    f"v_mov_b32 v2, {MODS_OLD:#x}",
    "v_sin_f16_e64 v2, v4 op_sel:[1,1] clamp",      # hi = clamp(-1.0) = 0.0, lo preserved
    f"v_mov_b32 v3, {MODS_OLD:#x}",
    "v_cos_f16_e64 v3, |v4| op_sel:[0,1]",          # hi = cos(0.125) ~= sqrt(2)/2, lo preserved
    "exp mrt0 v0, v1, v2, v3 done vm",
    "s_endpgm",
]

# In-place op_sel:[1,0] reads hi before overwriting lo; SDWA reads WORD_1 and writes WORD_1.
INPLACE_V0 = (f2h(0.25) << 16) | f2h(0.0)
INPLACE_OLD = 0x0000BEEF
INPLACE_LINES = [
    f"v_mov_b32 v0, {INPLACE_V0:#x}",
    "v_sin_f16_e64 v0, v0 op_sel:[1,0]",            # lo = sin(0.25) = 1.0, hi stays 0.25
    f"v_mov_b32 v1, {INPLACE_V0:#x}",
    "v_cos_f16_e64 v1, v1 op_sel:[0,1]",            # hi = cos(0.0) = 1.0, lo stays 0.0
    f"v_mov_b32 v4, {INPLACE_V0:#x}",
    f"v_mov_b32 v2, {INPLACE_OLD:#x}",
    "v_sin_f16_sdwa v2, v4 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
    f"v_mov_b32 v3, {INPLACE_OLD:#x}",
    "v_cos_f16_sdwa v3, v4 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0",
    "exp mrt0 v0, v1, v2, v3 done vm",
    "s_endpgm",
]


def positive_control_lines(op):
    valid_forms = [
        f"{op} v0, v1",
        f"{op}_e32 v0, v1",
        f"{op}_e64 v0, -|v1| op_sel:[1,1] clamp mul:2",
        f"{op} v0, |v1| op_sel:[1,0] div:2 clamp",
        f"{op}_sdwa v0, -v1 clamp mul:2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
    ]
    return [
        "v_mov_b32 v0, 0xaaaa3c00",
        "v_mov_b32 v1, 0x40003800",
        *valid_forms,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]


def translate_single(inst_line):
    lines = [
        "v_mov_b32 v0, 0xaaaa3c00",
        "v_mov_b32 v1, 0x40003800",
        "v_mov_b32 v2, 0x3c003c00",
        "s_mov_b32 s0, 0x40003800",
        "s_mov_b32 s1, 0x3c003c00",
        inst_line,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    return translate_lines("ps_unary_f16_single", lines)


class SinCosF16OpSelTests(unittest.TestCase):
    """Verify v_sin_f16 / v_cos_f16 honor op_sel:[shi, dhi] and preserve the other half."""

    def test_sin_cos_opsel_all_four_combinations(self):
        # sin(0.0) = 0.0, cos(0.0) = 1.0; sin(0.25) = 1.0, cos(0.25) = 0.0
        old = OPSEL_OLD
        expected = {
            ("v_sin_f16", 0, 0): (old & 0xFFFF0000) | f2h(0.0),
            ("v_sin_f16", 1, 0): (old & 0xFFFF0000) | f2h(1.0),
            ("v_sin_f16", 0, 1): (f2h(0.0) << 16) | (old & 0xFFFF),
            ("v_sin_f16", 1, 1): (f2h(1.0) << 16) | (old & 0xFFFF),
            ("v_cos_f16", 0, 0): (old & 0xFFFF0000) | f2h(1.0),
            ("v_cos_f16", 1, 0): (old & 0xFFFF0000) | f2h(0.0),
            ("v_cos_f16", 0, 1): (f2h(1.0) << 16) | (old & 0xFFFF),
            ("v_cos_f16", 1, 1): (f2h(0.0) << 16) | (old & 0xFFFF),
        }
        for op, suffix, name, lines in opsel_combo_programs():
            words = run_ps(lines, name)
            for idx, (shi, dhi) in enumerate(OPSEL_COMBOS):
                with self.subTest(op=op, suffix=suffix or "bare", shi=shi, dhi=dhi):
                    self.assertEqual(words[idx], expected[(op, shi, dhi)])

    def test_sin_cos_opsel_nontrivial_angles_and_mods(self):
        old = MODS_OLD
        words = run_ps(MODS_LINES, "ps_sincos_opsel_mods")
        self.assertEqual(words[0], (f2h(-1.0) << 16) | (old & 0xFFFF))
        self.assertEqual(words[1], (old & 0xFFFF0000) | f2h(1.0))
        self.assertEqual(words[2], (f2h(0.0) << 16) | (old & 0xFFFF))
        self.assertEqual(words[3] & 0xFFFF, old & 0xFFFF)
        self.assertAlmostEqual(h2f(words[3] >> 16), math.cos(2.0 * math.pi * 0.125), delta=2e-3)

    def test_sin_cos_opsel_inplace_and_sdwa(self):
        old = INPLACE_OLD
        words = run_ps(INPLACE_LINES, "ps_sincos_inplace_sdwa")
        self.assertEqual(words[0], (f2h(0.25) << 16) | f2h(1.0))
        self.assertEqual(words[1], (f2h(1.0) << 16) | f2h(0.0))
        self.assertEqual(words[2], (f2h(1.0) << 16) | (old & 0xFFFF))
        self.assertEqual(words[3], (f2h(1.0) << 16) | (old & 0xFFFF))

    @unittest.skipUnless(HAVE_SPIRV_TOOLS, "needs spirv-as and spirv-val")
    def test_sin_cos_opsel_modules_pass_spirv_val(self):
        programs = [(name, lines) for _, _, name, lines in opsel_combo_programs()]
        programs += [("ps_sincos_opsel_mods", MODS_LINES), ("ps_sincos_inplace_sdwa", INPLACE_LINES)]
        for name, lines in programs:
            with self.subTest(name=name):
                validate_spirv(translate_lines(name, lines))


class UnaryF16ValidationTests(unittest.TestCase):
    """Positive and negative operand/encoding/modifier validation across all 12 unary f16 ops."""

    def test_positive_controls_all_12_ops_translate(self):
        for op in UNARY_F16_OPS:
            with self.subTest(op=op):
                self.assertIn("OpEntryPoint", translate_lines(f"ps_pos_{op}", positive_control_lines(op)))

    @unittest.skipUnless(HAVE_SPIRV_TOOLS, "needs spirv-as and spirv-val")
    def test_positive_controls_all_12_ops_pass_spirv_val(self):
        for op in UNARY_F16_OPS:
            with self.subTest(op=op):
                validate_spirv(translate_lines(f"ps_pos_{op}", positive_control_lines(op)))

    def test_negative_validation_all_12_ops(self):
        invalid_templates = [
            # Arity
            ("arity_1", "{op} v0"),
            ("arity_3", "{op} v0, v1, v2"),
            # Destination validation
            ("dst_sgpr", "{op} s0, v1"),
            ("dst_vgpr64", "{op} v[0:1], v1"),
            ("dst_special", "{op} exec_lo, v1"),
            ("dst_neg", "{op} -v0, v1"),
            ("dst_abs", "{op} |v0|, v1"),
            ("dst_sext", "{op} sext(v0), v1"),
            # Source validation
            ("src_vgpr64", "{op} v0, v[1:2]"),
            ("src_sgpr64", "{op} v0, s[0:1]"),
            ("src_sext", "{op} v0, sext(v1)"),
            # _dpp rejected
            ("dpp_quad", "{op}_dpp v0, v1 quad_perm:[1,0,3,2]"),
            ("dpp8", "{op}_dpp v0, v1 dpp8:[0,1,2,3,4,5,6,7]"),
            # _e32 rejects source modifiers and all instruction modifiers
            ("e32_neg", "{op}_e32 v0, -v1"),
            ("e32_abs", "{op}_e32 v0, |v1|"),
            ("e32_clamp", "{op}_e32 v0, v1 clamp"),
            ("e32_opsel", "{op}_e32 v0, v1 op_sel:[1,0]"),
            ("e32_mul2", "{op}_e32 v0, v1 mul:2"),
            ("e32_div2", "{op}_e32 v0, v1 div:2"),
            # VOP3 (_e64 / bare) duplicate & invalid modifiers
            ("e64_dup_clamp", "{op}_e64 v0, v1 clamp clamp"),
            ("e64_dup_omod", "{op}_e64 v0, v1 mul:2 mul:2"),
            ("e64_conflict_omod", "{op}_e64 v0, v1 mul:2 div:2"),
            ("e64_bad_omod", "{op}_e64 v0, v1 mul:8"),
            ("e64_dup_opsel", "{op}_e64 v0, v1 op_sel:[0,0] op_sel:[1,1]"),
            ("e64_opsel_oob", "{op}_e64 v0, v1 op_sel:[2,0]"),
            ("e64_opsel_too_long", "{op}_e64 v0, v1 op_sel:[0,0,0]"),
            ("e64_opsel_empty", "{op}_e64 v0, v1 op_sel:[]"),
            ("e64_opsel_malformed", "{op}_e64 v0, v1 op_sel:bad"),
            ("e64_opsel_hi", "{op}_e64 v0, v1 op_sel_hi:[1,1]"),
            ("e64_neg_lo", "{op}_e64 v0, v1 neg_lo:[1,0]"),
            ("e64_neg_hi", "{op}_e64 v0, v1 neg_hi:[1,0]"),
            ("e64_sdwa_mod", "{op}_e64 v0, v1 src0_sel:WORD_1"),
            ("e64_unknown_mod", "{op}_e64 v0, v1 bogus_mod"),
            ("bare_opsel_hi", "{op} v0, v1 op_sel_hi:[1,1]"),
            ("bare_dup_clamp", "{op} v0, v1 clamp clamp"),
            ("bare_conflict_omod", "{op} v0, v1 mul:4 div:2"),
            # _sdwa invalid modifiers
            ("sdwa_opsel", "{op}_sdwa v0, v1 op_sel:[1,0] src0_sel:WORD_1"),
            ("sdwa_src1_sel", "{op}_sdwa v0, v1 src0_sel:WORD_0 src1_sel:WORD_1"),
            ("sdwa_bad_dst_sel", "{op}_sdwa v0, v1 dst_sel:BAD"),
            ("sdwa_bad_src0_sel", "{op}_sdwa v0, v1 src0_sel:BAD"),
            ("sdwa_bad_dst_unused", "{op}_sdwa v0, v1 dst_unused:BAD"),
            ("sdwa_dup_clamp", "{op}_sdwa v0, v1 clamp clamp src0_sel:WORD_0"),
            ("sdwa_dup_omod", "{op}_sdwa v0, v1 mul:2 div:2 src0_sel:WORD_0"),
            ("sdwa_dup_src0_sel", "{op}_sdwa v0, v1 src0_sel:WORD_0 src0_sel:WORD_1"),
            ("sdwa_sext_src", "{op}_sdwa v0, sext(v1) src0_sel:WORD_0"),
            ("sdwa_unknown_mod", "{op}_sdwa v0, v1 bogus_mod"),
        ]
        for op in UNARY_F16_OPS:
            for case_name, tmpl in invalid_templates:
                inst = tmpl.format(op=op)
                with self.subTest(op=op, case=case_name, inst=inst):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_single(inst)


if __name__ == "__main__":
    unittest.main()
