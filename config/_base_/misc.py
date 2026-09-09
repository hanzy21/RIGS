print_freq = 50
work_dir = None
load_from = None
max_epochs = 10000
max_iters_per_epoch = None  # 限制每个epoch的最大迭代次数，None表示不限制

# ================== training ========================
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=2e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35
