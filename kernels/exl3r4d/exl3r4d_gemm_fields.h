/* exl3r4d_gemm_fields.h -- the fields of exl3r4d's kernel argument block after its class records, included by
 * exl3r4d_dev.h into both blocks: Exl3r4dGemm (two classes) and Exl3r4dGemm4 (four, interface v6). One list, so the
 * two cannot drift apart and the two-class block's layout is what it was before the four-class one existed. No include
 * guard: it is a struct body, included once a block. */
    const void*    x;        /* gate_up: a [n_tok][x_pitch] f16|bf16 (token order);
                              * down: h [>= T][x_pitch] f16 (sorted order) */
    void*          y;        /* gate_up: h f16 [T][y_pitch]; down: y bf16 [T][y_pitch] */
    const int32_t* sorted;   /* [>= T] flat (token, slot) index of sorted row i, -1 dropped */
    const int32_t* off;      /* [n_expert + 1] expert e's rows are [off[e], off[e+1]) */
    void*          xr;       /* the prep kernel's rotated rows, f16 [T][xr_pitch] (scratch); null
                              * when the GEMM rotates its own rows (fused prep) */
    int64_t        x_pitch, y_pitch, xr_pitch;   /* elements */
    int32_t        T;        /* sorted rows = the output's rows */
    int32_t        n_tok;    /* rows of x (gate_up: tokens; down: >= T) */
    int32_t        n_expert, top_k, n_class;
    int32_t        y_cols;   /* the output's width: the wider class's */
    int32_t        x_bf16;   /* gate_up's a is bf16 (else f16) */
    int32_t        sk;       /* LDS split-K: k-slices a column group of waves */
    int32_t        chunk;    /* fused prep: rows a pass (what fits in LDS, <= 16); else 16 */
    int32_t        nslot;    /* grid.y: workgroups a column block (live experts + spares) */
    int32_t        h_rot;    /* v3 EPI 1: gate_up writes h rotated for down (dn_suh); down reads it so */
    int32_t        a_rot;    /* v4: gate_up's `a` is moe_exl3_prep_gu's xr (sorted rows, already rotated) */
    unsigned*      ctr;      /* D18 single launch (gate_up): this launch's counter block -- [0] tickets, [1] done,
                              * [2 + slot] slot ready -- zero at entry, reset to zero by the last workgroup; null:
                              * the prep kernel and the GEMM as two dispatches */
    int32_t        tpw;      /* split epilogue: 16-column tiles a GEMM workgroup owns (of one matrix) */
    float*         raw;      /* split epilogue (lever L8): the GEMM's raw f32 sums [T][raw_pitch] (scratch), which
                              * the finisher rotates, scales and writes out; null: the GEMM's own epilogue */
    int64_t        raw_pitch;
#if EXL3R4D_SUHC  /* (two fields: the argument block grows 16 B; gate_up_down's 744 B < 768) -- with it on */
    void*          suhc;     /* SUHC: this layer's resident copy of every expert's suh, suh.1 -- f16 [n_expert][2][K] --
                              * filled by the prep as it meets each expert; null: the planes, wherever they lie */
    unsigned*      suhc_ok;  /* SUHC: [n_expert] one bit a wave's slice of the copy written (0xFF: the whole) */
#endif
#if EXL3R4D_WGQ   /* (the arm's fields: the shipped argument block is unchanged while it is off) */
    unsigned*      wq;       /* WGQ (gate_up): this launch's counter block -- [0, nl) list cursors, [nl, 2 nl) active
                              * workgroups a WGP, [2 nl] done -- zero at entry, reset by the last workgroup; null: the grid */
    const int*     wq_map;   /* WGQ: HW_ID1 key (exl3r4d.hip, wgp_key) -> dense WGP id, or -1; probed at instance init */
    int32_t        wq_nl;    /* WGQ: WGPs (lists) */
#endif
