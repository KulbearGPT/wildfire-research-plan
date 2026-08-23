# Official released-weight manifest

The canonical manifest for the twelve official Res18-U-Net T=1 released
weights remains
[`reproductions/wsts_res18_unet_t1/official_weights_manifest.json`](../../reproductions/wsts_res18_unet_t1/official_weights_manifest.json).
It is pinned to Hugging Face Hub revision
`acf70a37394849f4ec8d108a51d6f4325a554d0a` and records the filename and byte
size of every fold weight. The cluster preflight intentionally does not
re-hash large files.

This directory intentionally does not duplicate those rows. Cluster transfer
and cache verification must consume the canonical tracked manifest above.
