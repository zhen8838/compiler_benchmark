# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def add(lv2: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(384)):
            for ax2_fused in T.vectorized(T.int64(1)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(lv2[v_ax0, v_ax1, v_ax2])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2])
                    T_add[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2] + T.float32(9.9999997473787516e-06)

    @T.prim_func
    def add1(lv39: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), lv40: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv39[v_ax0, v_ax1, v_ax2, v_ax3], lv40[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv39[v_ax0, v_ax1, v_ax2, v_ax3] + lv40[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def add2(lv45: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32"), attn_mask: T.Buffer((T.int64(1), T.int64(1), T.int64(384), T.int64(384)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(384)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv45[v_ax0, v_ax1, v_ax2, v_ax3], attn_mask[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = lv45[v_ax0, v_ax1, v_ax2, v_ax3] + attn_mask[v_ax0, T.int64(0), v_ax2, v_ax3]

    @T.prim_func
    def add3(lv: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), lv54: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(8192)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv[v_ax0, v_ax1, v_ax2], lv54[v_ax0, v_ax1, v_ax2])
                T.writes(T_add[v_ax0, v_ax1, v_ax2])
                T_add[v_ax0, v_ax1, v_ax2] = lv[v_ax0, v_ax1, v_ax2] + lv54[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def cast(hidden_in: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), compute: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(384), T.int64(8192)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(hidden_in[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = hidden_in[v_i0, v_i1, v_i2]

    @T.prim_func
    def cast1(lv15: T.Buffer((T.int64(1), T.int64(1), T.int64(384), T.int64(128)), "float32"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv15[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = lv15[v_i0, v_i1, v_i2, v_i3]

    @T.prim_func
    def cast2(lv47: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32"), compute: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(384)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv47[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = lv47[v_i0, v_i1, v_i2, v_i3]

    @T.prim_func
    def concatenate(lv28: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(64)), "float32"), lv29: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(64)), "float32"), T_concat: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(128)):
            with T.block("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv29[v_ax0, v_ax1, v_ax2, v_ax3 - T.int64(64)], lv28[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(64) <= v_ax3, lv29[v_ax0, v_ax1, v_ax2, v_ax3 - T.int64(64)], lv28[v_ax0, v_ax1, v_ax2, v_ax3])

    @T.prim_func
    def divide(lv4: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(384)):
            for ax2_fused in T.vectorized(T.int64(1)):
                with T.block("T_divide"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(lv4[v_ax0, v_ax1, v_ax2])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = T.float32(1.0) / lv4[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def divide1(lv44: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(384)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv44[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = lv44[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.088388349161020605)

    @T.prim_func
    def expand_dims(lv23: T.Buffer((T.int64(1), T.int64(384), T.int64(128)), "float32"), expand_dims: T.Buffer((T.int64(1), T.int64(1), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(128)):
            with T.block("expand_dims"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv23[v_i0, v_i2, v_i3])
                T.writes(expand_dims[v_i0, v_i1, v_i2, v_i3])
                expand_dims[v_i0, v_i1, v_i2, v_i3] = lv23[v_i0, v_i2, v_i3]

    @T.prim_func
    def matmul(lv8: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), B: T.Buffer((T.int64(8192), T.int64(8192)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for i1_0, i2_0, i0_1, i1_1, i2_1 in T.grid(T.int64(1), T.int64(8), T.int64(1), T.int64(2), T.int64(8)):
                for i0_2_init, i1_2_init, i2_2_init, i0_3_init, i1_3_init in T.grid(T.int64(1), T.int64(24), T.int64(8), T.int64(1), T.int64(8)):
                    for i2_3_fused_init in T.vectorized(T.int64(16)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2_init + i0_3_init)
                            v_i1 = T.axis.spatial(T.int64(384), i1_0 * T.int64(384) + i1_1 * T.int64(192) + i1_2_init * T.int64(8) + i1_3_init)
                            v_i2 = T.axis.spatial(T.int64(8192), i2_0 * T.int64(1024) + i2_1 * T.int64(128) + i2_2_init * T.int64(16) + i2_3_fused_init)
                            T.reads()
                            T.writes(matmul[v_i0, v_i1, v_i2])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                for k_0, i0_2, i1_2, i2_2, k_1, i0_3, i1_3 in T.grid(T.int64(1024), T.int64(1), T.int64(24), T.int64(8), T.int64(8), T.int64(1), T.int64(8)):
                    for i2_3_fused in T.vectorized(T.int64(16)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2 + i0_3)
                            v_i1 = T.axis.spatial(T.int64(384), i1_0 * T.int64(384) + i1_1 * T.int64(192) + i1_2 * T.int64(8) + i1_3)
                            v_i2 = T.axis.spatial(T.int64(8192), i2_0 * T.int64(1024) + i2_1 * T.int64(128) + i2_2 * T.int64(16) + i2_3_fused)
                            v_k = T.axis.reduce(T.int64(8192), k_0 * T.int64(8) + k_1)
                            T.reads(matmul[v_i0, v_i1, v_i2], lv8[v_i0, v_i1, v_k], B[v_k, v_i2])
                            T.writes(matmul[v_i0, v_i1, v_i2])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + lv8[v_i0, v_i1, v_k] * B[v_k, v_i2]

    @T.prim_func
    def matmul1(lv42: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), lv43: T.Buffer((T.int64(1), T.int64(64), T.int64(128), T.int64(384)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_global = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)))
        for i0_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(32), T.int64(6), T.int64(1)):
                for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i0_3_init, i1_3_init, i2_3_init, i3_3_init in T.grid(T.int64(1), T.int64(2), T.int64(32), T.int64(384), T.int64(1), T.int64(1), T.int64(2), T.int64(1)):
                    with T.block("matmul_init"):
                        v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2_init + i0_3_init)
                        v_i1 = T.axis.spatial(T.int64(64), i1_0 * T.int64(64) + i1_1 * T.int64(2) + i1_2_init + i1_3_init)
                        v_i2 = T.axis.spatial(T.int64(384), i2_0 * T.int64(384) + i2_1 * T.int64(64) + i2_2_init * T.int64(2) + i2_3_init)
                        v_i3 = T.axis.spatial(T.int64(384), i3_0 * T.int64(384) + i3_1 * T.int64(384) + i3_2_init + i3_3_init)
                        T.reads()
                        T.writes(matmul_global[v_i0, v_i1, v_i2, v_i3])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        matmul_global[v_i0, v_i1, v_i2, v_i3] = T.float32(0.0)
                for k_0, i0_2, i1_2, i2_2, i3_2, k_1, i0_3, i1_3, i2_3, i3_3 in T.grid(T.int64(16), T.int64(1), T.int64(2), T.int64(32), T.int64(384), T.int64(8), T.int64(1), T.int64(1), T.int64(2), T.int64(1)):
                    with T.block("matmul_update"):
                        v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2 + i0_3)
                        v_i1 = T.axis.spatial(T.int64(64), i1_0 * T.int64(64) + i1_1 * T.int64(2) + i1_2 + i1_3)
                        v_i2 = T.axis.spatial(T.int64(384), i2_0 * T.int64(384) + i2_1 * T.int64(64) + i2_2 * T.int64(2) + i2_3)
                        v_i3 = T.axis.spatial(T.int64(384), i3_0 * T.int64(384) + i3_1 * T.int64(384) + i3_2 + i3_3)
                        v_k = T.axis.reduce(T.int64(128), k_0 * T.int64(8) + k_1)
                        T.reads(matmul_global[v_i0, v_i1, v_i2, v_i3], lv42[v_i0, v_i1, v_i2, v_k], lv43[v_i0, v_i1, v_k, v_i3])
                        T.writes(matmul_global[v_i0, v_i1, v_i2, v_i3])
                        T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        matmul_global[v_i0, v_i1, v_i2, v_i3] = matmul_global[v_i0, v_i1, v_i2, v_i3] + lv42[v_i0, v_i1, v_i2, v_k] * lv43[v_i0, v_i1, v_k, v_i3]
                for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(64), T.int64(384)):
                    with T.block("matmul_global"):
                        v0 = T.axis.spatial(T.int64(1), ax0)
                        v1 = T.axis.spatial(T.int64(64), i1_1 * T.int64(2) + ax1)
                        v2 = T.axis.spatial(T.int64(384), i2_1 * T.int64(64) + ax2)
                        v3 = T.axis.spatial(T.int64(384), ax3)
                        T.reads(matmul_global[v0, v1, v2, v3])
                        T.writes(matmul[v0, v1, v2, v3])
                        matmul[v0, v1, v2, v3] = matmul_global[v0, v1, v2, v3]

    @T.prim_func
    def matmul2(lv49: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32"), lv50: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_global = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)))
        for i0_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}):
            for i1_0, i2_0, i3_0, i0_1, i1_1, i2_1, i3_1 in T.grid(T.int64(32), T.int64(6), T.int64(1), T.int64(1), T.int64(1), T.int64(4), T.int64(4)):
                for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i0_3_init, i1_3_init, i2_3_init in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(2), T.int64(1), T.int64(2), T.int64(4)):
                    for i3_3_fused_init in T.vectorized(T.int64(16)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2_init + i0_3_init)
                            v_i1 = T.axis.spatial(T.int64(64), i1_0 * T.int64(2) + i1_1 * T.int64(2) + i1_2_init * T.int64(2) + i1_3_init)
                            v_i2 = T.axis.spatial(T.int64(384), i2_0 * T.int64(64) + i2_1 * T.int64(16) + i2_2_init * T.int64(4) + i2_3_init)
                            v_i3 = T.axis.spatial(T.int64(128), i3_0 * T.int64(128) + i3_1 * T.int64(32) + i3_2_init * T.int64(16) + i3_3_fused_init)
                            T.reads()
                            T.writes(matmul_global[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            matmul_global[v_i0, v_i1, v_i2, v_i3] = T.float32(0.0)
                for k_0, i0_2, i1_2, i2_2, i3_2, k_1, i0_3, i1_3, i2_3 in T.grid(T.int64(6), T.int64(1), T.int64(1), T.int64(4), T.int64(2), T.int64(64), T.int64(1), T.int64(2), T.int64(4)):
                    for i3_3_fused in T.vectorized(T.int64(16)):
                        with T.block("matmul_update"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2 + i0_3)
                            v_i1 = T.axis.spatial(T.int64(64), i1_0 * T.int64(2) + i1_1 * T.int64(2) + i1_2 * T.int64(2) + i1_3)
                            v_i2 = T.axis.spatial(T.int64(384), i2_0 * T.int64(64) + i2_1 * T.int64(16) + i2_2 * T.int64(4) + i2_3)
                            v_i3 = T.axis.spatial(T.int64(128), i3_0 * T.int64(128) + i3_1 * T.int64(32) + i3_2 * T.int64(16) + i3_3_fused)
                            v_k = T.axis.reduce(T.int64(384), k_0 * T.int64(64) + k_1)
                            T.reads(matmul_global[v_i0, v_i1, v_i2, v_i3], lv49[v_i0, v_i1, v_i2, v_k], lv50[v_i0, v_i1, v_k, v_i3])
                            T.writes(matmul_global[v_i0, v_i1, v_i2, v_i3])
                            T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            matmul_global[v_i0, v_i1, v_i2, v_i3] = matmul_global[v_i0, v_i1, v_i2, v_i3] + lv49[v_i0, v_i1, v_i2, v_k] * lv50[v_i0, v_i1, v_k, v_i3]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(2), T.int64(16)):
                    for ax3_fused in T.vectorized(T.int64(32)):
                        with T.block("matmul_global"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(64), i1_0 * T.int64(2) + ax1)
                            v2 = T.axis.spatial(T.int64(384), i2_0 * T.int64(64) + i2_1 * T.int64(16) + ax2)
                            v3 = T.axis.spatial(T.int64(128), i3_1 * T.int64(32) + ax3_fused)
                            T.reads(matmul_global[v0, v1, v2, v3])
                            T.writes(matmul[v0, v1, v2, v3])
                            matmul[v0, v1, v2, v3] = matmul_global[v0, v1, v2, v3]

    @T.prim_func
    def matmul3(lv64: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), B: T.Buffer((T.int64(8192), T.int64(22016)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(384), T.int64(22016)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_global = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(22016)))
        for i0_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for i1_0, i2_0 in T.grid(T.int64(2), T.int64(2)):
                for i0_1, i1_1, i2_1 in T.grid(T.int64(1), T.int64(4), T.int64(4)):
                    for i0_2_init, i1_2_init, i2_2_init, i0_3_init, i1_3_init in T.grid(T.int64(1), T.int64(24), T.int64(86), T.int64(1), T.int64(2)):
                        for i2_3_fused_init in T.vectorized(T.int64(32)):
                            with T.block("matmul_init"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2_init + i0_3_init)
                                v_i1 = T.axis.spatial(T.int64(384), i1_0 * T.int64(192) + i1_1 * T.int64(48) + i1_2_init * T.int64(2) + i1_3_init)
                                v_i2 = T.axis.spatial(T.int64(22016), i2_0 * T.int64(11008) + i2_1 * T.int64(2752) + i2_2_init * T.int64(32) + i2_3_fused_init)
                                T.reads()
                                T.writes(matmul_global[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                                matmul_global[v_i0, v_i1, v_i2] = T.float32(0.0)
                    for k_0, i0_2, i1_2, i2_2, k_1, i0_3, i1_3 in T.grid(T.int64(1024), T.int64(1), T.int64(24), T.int64(86), T.int64(8), T.int64(1), T.int64(2)):
                        for i2_3_fused in T.vectorized(T.int64(32)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2 + i0_3)
                                v_i1 = T.axis.spatial(T.int64(384), i1_0 * T.int64(192) + i1_1 * T.int64(48) + i1_2 * T.int64(2) + i1_3)
                                v_i2 = T.axis.spatial(T.int64(22016), i2_0 * T.int64(11008) + i2_1 * T.int64(2752) + i2_2 * T.int64(32) + i2_3_fused)
                                v_k = T.axis.reduce(T.int64(8192), k_0 * T.int64(8) + k_1)
                                T.reads(matmul_global[v_i0, v_i1, v_i2], lv64[v_i0, v_i1, v_k], B[v_k, v_i2])
                                T.writes(matmul_global[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                                matmul_global[v_i0, v_i1, v_i2] = matmul_global[v_i0, v_i1, v_i2] + lv64[v_i0, v_i1, v_k] * B[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(192), T.int64(11008)):
                    with T.block("matmul_global"):
                        v0 = T.axis.spatial(T.int64(1), ax0)
                        v1 = T.axis.spatial(T.int64(384), i1_0 * T.int64(192) + ax1)
                        v2 = T.axis.spatial(T.int64(22016), i2_0 * T.int64(11008) + ax2)
                        T.reads(matmul_global[v0, v1, v2])
                        T.writes(matmul[v0, v1, v2])
                        matmul[v0, v1, v2] = matmul_global[v0, v1, v2]

    @T.prim_func
    def matmul4(lv69: T.Buffer((T.int64(1), T.int64(384), T.int64(22016)), "float32"), B: T.Buffer((T.int64(22016), T.int64(8192)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        matmul_global = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(8192)))
        for i0_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for i1_0, i2_0 in T.grid(T.int64(1), T.int64(1)):
                for i0_1, i1_1, i2_1 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    for i0_2_init, i1_2_init, i2_2_init, i0_3_init, i1_3_init in T.grid(T.int64(1), T.int64(64), T.int64(512), T.int64(1), T.int64(6)):
                        for i2_3_fused_init in T.vectorized(T.int64(16)):
                            with T.block("matmul_init"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2_init + i0_3_init)
                                v_i1 = T.axis.spatial(T.int64(384), i1_0 * T.int64(384) + i1_1 * T.int64(384) + i1_2_init * T.int64(6) + i1_3_init)
                                v_i2 = T.axis.spatial(T.int64(8192), i2_0 * T.int64(8192) + i2_1 * T.int64(8192) + i2_2_init * T.int64(16) + i2_3_fused_init)
                                T.reads()
                                T.writes(matmul_global[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                                matmul_global[v_i0, v_i1, v_i2] = T.float32(0.0)
                    for k_0, i0_2, i1_2, i2_2, k_1, i0_3, i1_3 in T.grid(T.int64(512), T.int64(1), T.int64(64), T.int64(512), T.int64(43), T.int64(1), T.int64(6)):
                        for i2_3_fused in T.vectorized(T.int64(16)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_0 + i0_1 + i0_2 + i0_3)
                                v_i1 = T.axis.spatial(T.int64(384), i1_0 * T.int64(384) + i1_1 * T.int64(384) + i1_2 * T.int64(6) + i1_3)
                                v_i2 = T.axis.spatial(T.int64(8192), i2_0 * T.int64(8192) + i2_1 * T.int64(8192) + i2_2 * T.int64(16) + i2_3_fused)
                                v_k = T.axis.reduce(T.int64(22016), k_0 * T.int64(43) + k_1)
                                T.reads(matmul_global[v_i0, v_i1, v_i2], lv69[v_i0, v_i1, v_k], B[v_k, v_i2])
                                T.writes(matmul_global[v_i0, v_i1, v_i2])
                                T.block_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                                matmul_global[v_i0, v_i1, v_i2] = matmul_global[v_i0, v_i1, v_i2] + lv69[v_i0, v_i1, v_k] * B[v_k, v_i2]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(8192)):
                    with T.block("matmul_global"):
                        v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                        T.reads(matmul_global[v0, v1, v2])
                        T.writes(matmul[v0, v1, v2])
                        matmul[v0, v1, v2] = matmul_global[v0, v1, v2]

    @T.prim_func
    def mean(lv1: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        lv1_red = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1)))
        lv1_red_rf = T.alloc_buffer((T.int64(1), T.int64(384), T.int64(1), T.int64(32)))
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1)):
            for k2_1_fused_init in T.vectorized(T.int64(32)):
                with T.block("lv1_red_rf_init"):
                    vk2_1, v_ax0, v_ax1, v_ax2 = T.axis.remap("SSSS", [k2_1_fused_init, ax0, ax1, ax2])
                    T.reads()
                    T.writes(lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1])
                    lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1] = T.float32(0.0)
            for k2_0 in range(T.int64(256)):
                for k2_1_fused in T.vectorized(T.int64(32)):
                    with T.block("lv1_red_rf_update"):
                        vk2_1, v_ax0, v_ax1, v_ax2, vk2_0 = T.axis.remap("SSSSR", [k2_1_fused, ax0, ax1, ax2, k2_0])
                        T.reads(lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1], lv1[v_ax0, v_ax1, vk2_0 * T.int64(32) + vk2_1])
                        T.writes(lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1])
                        lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1] = lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1] + lv1[v_ax0, v_ax1, vk2_0 * T.int64(32) + vk2_1]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(1)):
            with T.block("lv1_red_init"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads()
                T.writes(lv1_red[v_ax0, v_ax1, v_ax2])
                lv1_red[v_ax0, v_ax1, v_ax2] = T.float32(0.0)
            for k2_1 in range(T.int64(32)):
                with T.block("lv1_red_update"):
                    vk2_1, v_ax0, v_ax1, v_ax2 = T.axis.remap("RSSS", [k2_1, ax0, ax1, ax2])
                    T.reads(lv1_red[v_ax0, v_ax1, v_ax2], lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1])
                    T.writes(lv1_red[v_ax0, v_ax1, v_ax2])
                    lv1_red[v_ax0, v_ax1, v_ax2] = lv1_red[v_ax0, v_ax1, v_ax2] + lv1_red_rf[v_ax0, v_ax1, v_ax2, vk2_1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(384)):
            for ax2_fused in T.vectorized(T.int64(1)):
                with T.block("T_divide"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(lv1_red[v_ax0, v_ax1, v_ax2])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = lv1_red[v_ax0, v_ax1, v_ax2] * T.float32(0.0001220703125)

    @T.prim_func
    def multiply(lv: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), lv5: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(8192)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv[v_ax0, v_ax1, v_ax2], lv5[v_ax0, v_ax1, T.int64(0)])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = lv[v_ax0, v_ax1, v_ax2] * lv5[v_ax0, v_ax1, T.int64(0)]

    @T.prim_func
    def multiply1(A: T.Buffer((T.int64(8192),), "float32"), lv7: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(8192)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax2], lv7[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = A[v_ax2] * lv7[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def multiply2(lv25: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), lv26: T.Buffer((T.int64(1), T.int64(1), T.int64(384), T.int64(128)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(128)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv25[v_ax0, v_ax1, v_ax2, v_ax3], lv26[v_ax0, T.int64(0), v_ax2, v_ax3])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = lv25[v_ax0, v_ax1, v_ax2, v_ax3] * lv26[v_ax0, T.int64(0), v_ax2, v_ax3]

    @T.prim_func
    def multiply3(lv65: T.Buffer((T.int64(1), T.int64(384), T.int64(22016)), "float32"), lv66: T.Buffer((T.int64(1), T.int64(384), T.int64(22016)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(384), T.int64(22016)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(22016)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv65[v_ax0, v_ax1, v_ax2], lv66[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = lv65[v_ax0, v_ax1, v_ax2] * lv66[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def power(lv: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), T_power: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(8192)):
            with T.block("T_power"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv[v_ax0, v_ax1, v_ax2])
                T.writes(T_power[v_ax0, v_ax1, v_ax2])
                T_power[v_ax0, v_ax1, v_ax2] = T.pow(lv[v_ax0, v_ax1, v_ax2], T.float32(2.0))

    @T.prim_func
    def reshape(lv9: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(384), T.int64(64), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(384), T.int64(64), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv9[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(8192) + v_ax1) % T.int64(384), (v_ax2 * T.int64(128) + v_ax3) % T.int64(8192)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = lv9[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(8192) + v_ax1) % T.int64(384), (v_ax2 * T.int64(128) + v_ax3) % T.int64(8192)]

    @T.prim_func
    def reshape1(lv52: T.Buffer((T.int64(1), T.int64(384), T.int64(64), T.int64(128)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(384), T.int64(8192)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(8192)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv52[T.int64(0), (v_ax2 // T.int64(8192) + v_ax1) % T.int64(384), v_ax2 % T.int64(8192) // T.int64(128), v_ax2 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = lv52[T.int64(0), (v_ax2 // T.int64(8192) + v_ax1) % T.int64(384), v_ax2 % T.int64(8192) // T.int64(128), v_ax2 % T.int64(128)]

    @T.prim_func
    def softmax(lv46: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(384)))
        T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(384)))
        T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(384)))
        T_softmax_expsum_rf = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(48)))
        T_softmax_maxelem_rf = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(2)))
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(64), T.int64(384)):
            for k_1_fused_init in T.vectorized(T.int64(2)):
                with T.block("T_softmax_maxelem_rf_init"):
                    vk_1, v_i0, v_i1, v_i2 = T.axis.remap("SSSS", [k_1_fused_init, i0, i1, i2])
                    T.reads()
                    T.writes(T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1])
                    T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1] = T.float32(-340282346638528859811704183484516925440.0)
            for k_0 in range(T.int64(192)):
                for k_1_fused in T.vectorized(T.int64(2)):
                    with T.block("T_softmax_maxelem_rf_update"):
                        vk_1, v_i0, v_i1, v_i2, vk_0 = T.axis.remap("SSSSR", [k_1_fused, i0, i1, i2, k_0])
                        T.reads(T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1], lv46[v_i0, v_i1, v_i2, vk_0 * T.int64(2) + vk_1])
                        T.writes(T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1])
                        T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1] = T.max(T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1], lv46[v_i0, v_i1, v_i2, vk_0 * T.int64(2) + vk_1])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(64), T.int64(384)):
            for ax1_init, ax2_init, ax3_init in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                with T.block("T_softmax_maxelem_init"):
                    v_i0 = T.axis.spatial(T.int64(1), ax1_init)
                    v_i1 = T.axis.spatial(T.int64(64), i1 + ax2_init)
                    v_i2 = T.axis.spatial(T.int64(384), i2 + ax3_init)
                    T.reads()
                    T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-340282346638528859811704183484516925440.0)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1)):
                with T.block("T_softmax_maxelem_update"):
                    vk_1, v_i0 = T.axis.remap("RS", [ax0, ax1])
                    v_i1 = T.axis.spatial(T.int64(64), i1 + ax2)
                    v_i2 = T.axis.spatial(T.int64(384), i2 + ax3)
                    T.reads(T_softmax_maxelem[v_i0, v_i1, v_i2], T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1])
                    T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], T_softmax_maxelem_rf[v_i0, v_i1, v_i2, vk_1])
            for i3 in range(T.int64(384)):
                with T.block("T_softmax_exp"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(lv46[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                    T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(lv46[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(64), T.int64(384)):
            for k_1_fused_init in T.vectorized(T.int64(48)):
                with T.block("T_softmax_expsum_rf_init"):
                    vk_1, v_i0, v_i1, v_i2 = T.axis.remap("SSSS", [k_1_fused_init, i0, i1, i2])
                    T.reads()
                    T.writes(T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1])
                    T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1] = T.float32(0.0)
            for k_0 in range(T.int64(8)):
                for k_1_fused in T.vectorized(T.int64(48)):
                    with T.block("T_softmax_expsum_rf_update"):
                        vk_1, v_i0, v_i1, v_i2, vk_0 = T.axis.remap("SSSSR", [k_1_fused, i0, i1, i2, k_0])
                        T.reads(T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1], T_softmax_exp[v_i0, v_i1, v_i2, vk_0 * T.int64(48) + vk_1])
                        T.writes(T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1])
                        T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1] = T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1] + T_softmax_exp[v_i0, v_i1, v_i2, vk_0 * T.int64(48) + vk_1]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(64), T.int64(384)):
            with T.block("T_softmax_expsum_init"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads()
                T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0.0)
            for k_1 in range(T.int64(48)):
                with T.block("T_softmax_expsum_update"):
                    vk_1, v_i0, v_i1, v_i2 = T.axis.remap("RSSS", [k_1, i0, i1, i2])
                    T.reads(T_softmax_expsum[v_i0, v_i1, v_i2], T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1])
                    T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                    T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_expsum_rf[v_i0, v_i1, v_i2, vk_1]
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(384)):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                T.block_attr({"axis": 3})
                T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

    @T.prim_func
    def strided_slice(A: T.Buffer((T.int64(1), T.int64(1), T.int64(2048), T.int64(128)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(1), T.int64(1), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(384), T.int64(128)):
            with T.block("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def strided_slice1(lv25: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(64), T.int64(384)):
            for ax3_fused in T.vectorized(T.int64(64)):
                with T.block("T_strided_slice_with_axes"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3_fused])
                    T.reads(lv25[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(64)])
                    T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = lv25[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(64)]

    @T.prim_func
    def strided_slice2(lv25: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(64), T.int64(384)):
            for ax3_fused in T.vectorized(T.int64(64)):
                with T.block("T_strided_slice_with_axes"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3_fused])
                    T.reads(lv25[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = lv25[v_ax0, v_ax1, v_ax2, v_ax3]

    @T.prim_func
    def take(A: T.Buffer((T.int64(1), T.int64(1), T.int64(384), T.int64(128)), "float32"), B: T.Buffer((), "int64"), T_take: T.Buffer((T.int64(1), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(128)):
            with T.block("T_take"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                T.writes(T_take[v_ax0, v_ax1, v_ax2])
                T_take[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def take1(lv18: T.Buffer((T.int64(1), T.int64(384), T.int64(128)), "float32"), B: T.Buffer((), "int64"), T_take: T.Buffer((T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(384), T.int64(128)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv18[T.int64(0), v_ax0, v_ax1])
                T.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = lv18[T.int64(0), v_ax0, v_ax1]

    @T.prim_func
    def take2(lv19: T.Buffer((T.int64(384), T.int64(128)), "float32"), position_ids: T.Buffer((T.int64(1), T.int64(384)), "int64"), T_take: T.Buffer((T.int64(1), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(384), T.int64(128)):
            with T.block("T_take"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv19[T.min(T.max(T.int64(0), position_ids[v_ax0, v_ax1]), T.int64(383)), v_ax2], position_ids[v_ax0, v_ax1])
                T.writes(T_take[v_ax0, v_ax1, v_ax2])
                T_take[v_ax0, v_ax1, v_ax2] = lv19[T.min(T.max(T.int64(0), position_ids[v_ax0, v_ax1]), T.int64(383)), v_ax2]

    @T.prim_func
    def tir_negative(lv27: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(64)), "float32"), compute: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(64), T.int64(384)):
            for i3_fused in T.vectorized(T.int64(64)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3_fused])
                    T.reads(lv27[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = lv27[v_i0, v_i1, v_i2, v_i3] * T.float32(-1.0)

    @T.prim_func
    def tir_sigmoid(lv65: T.Buffer((T.int64(1), T.int64(384), T.int64(22016)), "float32"), compute: T.Buffer((T.int64(1), T.int64(384), T.int64(22016)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(384), T.int64(22016)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv65[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.sigmoid(lv65[v_i0, v_i1, v_i2])

    @T.prim_func
    def tir_sqrt(lv3: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32"), compute: T.Buffer((T.int64(1), T.int64(384), T.int64(1)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(384)):
            for i2_fused in T.vectorized(T.int64(1)):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(lv3[v_i0, v_i1, v_i2])
                    T.writes(compute[v_i0, v_i1, v_i2])
                    compute[v_i0, v_i1, v_i2] = T.sqrt(lv3[v_i0, v_i1, v_i2])

    @T.prim_func
    def transpose(lv10: T.Buffer((T.int64(1), T.int64(384), T.int64(64), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(384), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv10[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = lv10[v_ax0, v_ax2, v_ax1, v_ax3]

    @T.prim_func
    def transpose1(lv41: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(64), T.int64(128), T.int64(384)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(128), T.int64(384)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv41[v_ax0, v_ax1, v_ax3, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = lv41[v_ax0, v_ax1, v_ax3, v_ax2]

    @T.prim_func
    def transpose2(lv51: T.Buffer((T.int64(1), T.int64(64), T.int64(384), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(384), T.int64(64), T.int64(128)), "float32")):
        T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(384), T.int64(64), T.int64(128)):
            with T.block("T_transpose"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(lv51[v_ax0, v_ax2, v_ax1, v_ax3])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = lv51[v_ax0, v_ax2, v_ax1, v_ax3]

    @R.function
    def main(hidden_in: R.Tensor((1, 384, 8192), dtype="float32"), attn_mask: R.Tensor((1, 1, 384, 384), dtype="float32"), position_ids: R.Tensor((1, 384), dtype="int64")) -> R.Tensor((1, 384, 8192), dtype="float32"):
        R.func_attr({"num_input": 3, "relax.force_pure": True})
        cls = Module
        shape_heap: R.Object = R.null_value()
        R.call_packed("vm.builtin.check_tensor_info", hidden_in, R.prim_value(3), R.dtype("float32"), R.str("ErrorContext(fn=main, loc=param[0], param=hidden_in, annotation=R.Tensor((1, 384, 8192), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        R.call_packed("vm.builtin.check_tensor_info", attn_mask, R.prim_value(4), R.dtype("float32"), R.str("ErrorContext(fn=main, loc=param[1], param=attn_mask, annotation=R.Tensor((1, 1, 384, 384), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        R.call_packed("vm.builtin.check_tensor_info", position_ids, R.prim_value(2), R.dtype("int64"), R.str("ErrorContext(fn=main, loc=param[2], param=position_ids, annotation=R.Tensor((1, 384), dtype=\"int64\")) "), sinfo_args=(R.Tuple,))
        R.call_packed("vm.builtin.match_shape", hidden_in, shape_heap, R.prim_value(3), R.prim_value(0), R.prim_value(1), R.prim_value(0), R.prim_value(384), R.prim_value(0), R.prim_value(8192), R.str("ErrorContext(fn=main, loc=param[0], param=hidden_in, annotation=R.Tensor((1, 384, 8192), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        R.call_packed("vm.builtin.match_shape", attn_mask, shape_heap, R.prim_value(4), R.prim_value(0), R.prim_value(1), R.prim_value(0), R.prim_value(1), R.prim_value(0), R.prim_value(384), R.prim_value(0), R.prim_value(384), R.str("ErrorContext(fn=main, loc=param[1], param=attn_mask, annotation=R.Tensor((1, 1, 384, 384), dtype=\"float32\")) "), sinfo_args=(R.Tuple,))
        R.call_packed("vm.builtin.match_shape", position_ids, shape_heap, R.prim_value(2), R.prim_value(0), R.prim_value(1), R.prim_value(0), R.prim_value(384), R.str("ErrorContext(fn=main, loc=param[2], param=position_ids, annotation=R.Tensor((1, 384), dtype=\"int64\")) "), sinfo_args=(R.Tuple,))
        lv: R.Tensor((1, 384, 8192), dtype="float32") = R.call_packed("vm.builtin.reshape", hidden_in, R.shape([1, 384, 8192]), sinfo_args=(R.Tensor((1, 384, 8192), dtype="float32"),))
        storage: R.Object = R.vm.alloc_storage(R.shape([12582912]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.power(lv, alloc)
        storage1: R.Object = R.vm.alloc_storage(R.shape([1536]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc1: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        cls.mean(alloc, alloc1)
        R.vm.kill_object(alloc)
        storage2: R.Object = R.vm.alloc_storage(R.shape([1536]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc2: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        cls.add(alloc1, alloc2)
        R.vm.kill_object(alloc1)
        alloc3: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        cls.tir_sqrt(alloc2, alloc3)
        R.vm.kill_object(alloc2)
        alloc4: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        cls.divide(alloc3, alloc4)
        R.vm.kill_object(alloc3)
        alloc5: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.multiply(lv, alloc4, alloc5)
        R.vm.kill_object(alloc4)
        lv7: R.Tensor((1, 384, 8192), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc5, R.shape([1, 384, 8192]), sinfo_args=(R.Tensor((1, 384, 8192), dtype="float32"),))
        R.vm.kill_object(alloc5)
        storage3: R.Object = R.vm.alloc_storage(R.shape([12582912]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc6: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.multiply1(metadata["relax.expr.Constant"][0], lv7, alloc6)
        R.vm.kill_object(lv7)
        alloc7: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.matmul(alloc6, metadata["relax.expr.Constant"][1], alloc7)
        lv10: R.Tensor((1, 384, 64, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc7, R.shape([1, 384, 64, 128]), sinfo_args=(R.Tensor((1, 384, 64, 128), dtype="float32"),))
        R.vm.kill_object(alloc7)
        storage4: R.Object = R.vm.alloc_storage(R.shape([33816576]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc8: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.matmul(alloc6, metadata["relax.expr.Constant"][2], alloc8)
        lv12: R.Tensor((1, 384, 64, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc8, R.shape([1, 384, 64, 128]), sinfo_args=(R.Tensor((1, 384, 64, 128), dtype="float32"),))
        R.vm.kill_object(alloc8)
        storage5: R.Object = R.vm.alloc_storage(R.shape([12582912]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc9: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.matmul(alloc6, metadata["relax.expr.Constant"][3], alloc9)
        R.vm.kill_object(alloc6)
        lv14: R.Tensor((1, 384, 64, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc9, R.shape([1, 384, 64, 128]), sinfo_args=(R.Tensor((1, 384, 64, 128), dtype="float32"),))
        R.vm.kill_object(alloc9)
        storage6: R.Object = R.vm.alloc_storage(R.shape([196608]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc10: R.Tensor((1, 1, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([1, 1, 384, 128]), R.dtype("float32"))
        cls.strided_slice(metadata["relax.expr.Constant"][4], alloc10)
        storage7: R.Object = R.vm.alloc_storage(R.shape([196608]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc11: R.Tensor((1, 1, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage7, R.prim_value(0), R.shape([1, 1, 384, 128]), R.dtype("float32"))
        R.vm.kill_object(storage7)
        cls.strided_slice(metadata["relax.expr.Constant"][5], alloc11)
        lv17: R.Tensor((1, 1, 384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc10, R.shape([1, 1, 384, 128]), sinfo_args=(R.Tensor((1, 1, 384, 128), dtype="float32"),))
        R.vm.kill_object(alloc10)
        lv18: R.Tensor((1, 384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", lv17, R.shape([1, 384, 128]), sinfo_args=(R.Tensor((1, 384, 128), dtype="float32"),))
        R.vm.kill_object(lv17)
        lv19: R.Tensor((384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", lv18, R.shape([384, 128]), sinfo_args=(R.Tensor((384, 128), dtype="float32"),))
        R.vm.kill_object(lv18)
        lv20: R.Tensor((1, 1, 384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc11, R.shape([1, 1, 384, 128]), sinfo_args=(R.Tensor((1, 1, 384, 128), dtype="float32"),))
        R.vm.kill_object(alloc11)
        lv21: R.Tensor((1, 384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", lv20, R.shape([1, 384, 128]), sinfo_args=(R.Tensor((1, 384, 128), dtype="float32"),))
        R.vm.kill_object(lv20)
        lv22: R.Tensor((384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", lv21, R.shape([384, 128]), sinfo_args=(R.Tensor((384, 128), dtype="float32"),))
        R.vm.kill_object(lv21)
        storage8: R.Object = R.vm.alloc_storage(R.shape([196608]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc12: R.Tensor((1, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage8, R.prim_value(0), R.shape([1, 384, 128]), R.dtype("float32"))
        R.vm.kill_object(storage8)
        cls.take2(lv19, position_ids, alloc12)
        R.vm.kill_object(lv19)
        alloc13: R.Tensor((1, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([1, 384, 128]), R.dtype("float32"))
        R.vm.kill_object(storage6)
        cls.take2(lv22, position_ids, alloc13)
        R.vm.kill_object(lv22)
        alloc14: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.transpose(lv10, alloc14)
        R.vm.kill_object(lv10)
        lv26: R.Tensor((1, 1, 384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc12, R.shape([1, 1, 384, 128]), sinfo_args=(R.Tensor((1, 1, 384, 128), dtype="float32"),))
        R.vm.kill_object(alloc12)
        alloc15: R.Tensor((1, 64, 384, 64), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 64, 384, 64]), R.dtype("float32"))
        cls.strided_slice1(alloc14, alloc15)
        storage9: R.Object = R.vm.alloc_storage(R.shape([6291456]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc16: R.Tensor((1, 64, 384, 64), dtype="float32") = R.vm.alloc_tensor(storage9, R.prim_value(0), R.shape([1, 64, 384, 64]), R.dtype("float32"))
        cls.tir_negative(alloc15, alloc16)
        R.vm.kill_object(alloc15)
        alloc17: R.Tensor((1, 64, 384, 64), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 64, 384, 64]), R.dtype("float32"))
        cls.strided_slice2(alloc14, alloc17)
        storage10: R.Object = R.vm.alloc_storage(R.shape([37748736]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc18: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.concatenate(alloc16, alloc17, alloc18)
        R.vm.kill_object(alloc16)
        R.vm.kill_object(alloc17)
        lv31: R.Tensor((1, 1, 384, 128), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc13, R.shape([1, 1, 384, 128]), sinfo_args=(R.Tensor((1, 1, 384, 128), dtype="float32"),))
        R.vm.kill_object(alloc13)
        alloc19: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.multiply2(alloc14, lv26, alloc19)
        R.vm.kill_object(alloc14)
        alloc20: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.multiply2(alloc18, lv31, alloc20)
        R.vm.kill_object(alloc18)
        alloc21: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.transpose(lv12, alloc21)
        R.vm.kill_object(lv12)
        alloc22: R.Tensor((1, 64, 384, 64), dtype="float32") = R.vm.alloc_tensor(storage9, R.prim_value(0), R.shape([1, 64, 384, 64]), R.dtype("float32"))
        cls.strided_slice1(alloc21, alloc22)
        alloc23: R.Tensor((1, 64, 384, 64), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([1, 64, 384, 64]), R.dtype("float32"))
        cls.tir_negative(alloc22, alloc23)
        R.vm.kill_object(alloc22)
        alloc24: R.Tensor((1, 64, 384, 64), dtype="float32") = R.vm.alloc_tensor(storage9, R.prim_value(0), R.shape([1, 64, 384, 64]), R.dtype("float32"))
        R.vm.kill_object(storage9)
        cls.strided_slice2(alloc21, alloc24)
        storage11: R.Object = R.vm.alloc_storage(R.shape([37748736]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc25: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage11, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.concatenate(alloc23, alloc24, alloc25)
        R.vm.kill_object(alloc23)
        R.vm.kill_object(alloc24)
        alloc26: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.multiply2(alloc21, lv26, alloc26)
        R.vm.kill_object(lv26)
        R.vm.kill_object(alloc21)
        alloc27: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.multiply2(alloc25, lv31, alloc27)
        R.vm.kill_object(lv31)
        R.vm.kill_object(alloc25)
        alloc28: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage11, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.add1(alloc26, alloc27, alloc28)
        R.vm.kill_object(alloc26)
        R.vm.kill_object(alloc27)
        alloc29: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.add1(alloc19, alloc20, alloc29)
        R.vm.kill_object(alloc19)
        R.vm.kill_object(alloc20)
        alloc30: R.Tensor((1, 64, 128, 384), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 64, 128, 384]), R.dtype("float32"))
        cls.transpose1(alloc28, alloc30)
        R.vm.kill_object(alloc28)
        alloc31: R.Tensor((1, 64, 384, 384), dtype="float32") = R.vm.alloc_tensor(storage11, R.prim_value(0), R.shape([1, 64, 384, 384]), R.dtype("float32"))
        cls.matmul1(alloc29, alloc30, alloc31)
        R.vm.kill_object(alloc29)
        R.vm.kill_object(alloc30)
        alloc32: R.Tensor((1, 64, 384, 384), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 64, 384, 384]), R.dtype("float32"))
        cls.divide1(alloc31, alloc32)
        R.vm.kill_object(alloc31)
        alloc33: R.Tensor((1, 64, 384, 384), dtype="float32") = R.vm.alloc_tensor(storage11, R.prim_value(0), R.shape([1, 64, 384, 384]), R.dtype("float32"))
        cls.add2(alloc32, attn_mask, alloc33)
        R.vm.kill_object(alloc32)
        alloc34: R.Tensor((1, 64, 384, 384), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 64, 384, 384]), R.dtype("float32"))
        cls.softmax(alloc33, alloc34)
        R.vm.kill_object(alloc33)
        lv48: R.Tensor((1, 64, 384, 384), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc34, R.shape([1, 64, 384, 384]), sinfo_args=(R.Tensor((1, 64, 384, 384), dtype="float32"),))
        R.vm.kill_object(alloc34)
        lv49: R.Tensor((1, 64, 384, 384), dtype="float32") = R.call_packed("vm.builtin.reshape", lv48, R.shape([1, 64, 384, 384]), sinfo_args=(R.Tensor((1, 64, 384, 384), dtype="float32"),))
        R.vm.kill_object(lv48)
        alloc35: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.transpose(lv14, alloc35)
        R.vm.kill_object(lv14)
        alloc36: R.Tensor((1, 64, 384, 128), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1, 64, 384, 128]), R.dtype("float32"))
        cls.matmul2(lv49, alloc35, alloc36)
        R.vm.kill_object(lv49)
        R.vm.kill_object(alloc35)
        alloc37: R.Tensor((1, 384, 64, 128), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([1, 384, 64, 128]), R.dtype("float32"))
        cls.transpose2(alloc36, alloc37)
        R.vm.kill_object(alloc36)
        lv53: R.Tensor((1, 384, 8192), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc37, R.shape([1, 384, 8192]), sinfo_args=(R.Tensor((1, 384, 8192), dtype="float32"),))
        R.vm.kill_object(alloc37)
        alloc38: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.matmul(lv53, metadata["relax.expr.Constant"][6], alloc38)
        R.vm.kill_object(lv53)
        alloc39: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        R.vm.kill_object(storage)
        cls.add3(lv, alloc38, alloc39)
        R.vm.kill_object(lv)
        R.vm.kill_object(alloc38)
        lv56: R.Tensor((1, 384, 8192), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc39, R.shape([1, 384, 8192]), sinfo_args=(R.Tensor((1, 384, 8192), dtype="float32"),))
        R.vm.kill_object(alloc39)
        alloc40: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.power(lv56, alloc40)
        alloc41: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        cls.mean(alloc40, alloc41)
        R.vm.kill_object(alloc40)
        alloc42: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        cls.add(alloc41, alloc42)
        R.vm.kill_object(alloc41)
        alloc43: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        R.vm.kill_object(storage1)
        cls.tir_sqrt(alloc42, alloc43)
        R.vm.kill_object(alloc42)
        alloc44: R.Tensor((1, 384, 1), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1, 384, 1]), R.dtype("float32"))
        R.vm.kill_object(storage2)
        cls.divide(alloc43, alloc44)
        R.vm.kill_object(alloc43)
        alloc45: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        cls.multiply(lv56, alloc44, alloc45)
        R.vm.kill_object(alloc44)
        lv63: R.Tensor((1, 384, 8192), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc45, R.shape([1, 384, 8192]), sinfo_args=(R.Tensor((1, 384, 8192), dtype="float32"),))
        R.vm.kill_object(alloc45)
        alloc46: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        R.vm.kill_object(storage5)
        cls.multiply1(metadata["relax.expr.Constant"][0], lv63, alloc46)
        R.vm.kill_object(lv63)
        alloc47: R.Tensor((1, 384, 22016), dtype="float32") = R.vm.alloc_tensor(storage11, R.prim_value(0), R.shape([1, 384, 22016]), R.dtype("float32"))
        cls.matmul3(alloc46, metadata["relax.expr.Constant"][7], alloc47)
        alloc48: R.Tensor((1, 384, 22016), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 384, 22016]), R.dtype("float32"))
        cls.tir_sigmoid(alloc47, alloc48)
        alloc49: R.Tensor((1, 384, 22016), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([1, 384, 22016]), R.dtype("float32"))
        R.vm.kill_object(storage4)
        cls.multiply3(alloc47, alloc48, alloc49)
        R.vm.kill_object(alloc47)
        R.vm.kill_object(alloc48)
        alloc50: R.Tensor((1, 384, 22016), dtype="float32") = R.vm.alloc_tensor(storage11, R.prim_value(0), R.shape([1, 384, 22016]), R.dtype("float32"))
        R.vm.kill_object(storage11)
        cls.matmul3(alloc46, metadata["relax.expr.Constant"][8], alloc50)
        R.vm.kill_object(alloc46)
        alloc51: R.Tensor((1, 384, 22016), dtype="float32") = R.vm.alloc_tensor(storage10, R.prim_value(0), R.shape([1, 384, 22016]), R.dtype("float32"))
        R.vm.kill_object(storage10)
        cls.multiply3(alloc49, alloc50, alloc51)
        R.vm.kill_object(alloc49)
        R.vm.kill_object(alloc50)
        alloc52: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        R.vm.kill_object(storage3)
        cls.matmul4(alloc51, metadata["relax.expr.Constant"][9], alloc52)
        R.vm.kill_object(alloc51)
        storage_1: R.Object = R.vm.alloc_storage(R.shape([12582912]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc53: R.Tensor((1, 384, 8192), dtype="float32") = R.vm.alloc_tensor(storage_1, R.prim_value(0), R.shape([1, 384, 8192]), R.dtype("float32"))
        R.vm.kill_object(storage_1)
        cls.add3(lv56, alloc52, alloc53)
        R.vm.kill_object(lv56)
        R.vm.kill_object(alloc52)
        return alloc53

# Metadata omitted. Use show_meta=True in script() method to show it.