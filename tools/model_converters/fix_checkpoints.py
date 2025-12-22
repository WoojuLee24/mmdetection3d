import torch
from pathlib import Path

# 🔧 수정 전/후 경로 설정
OLD_CKPT = Path("/ws/external/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth")
NEW_CKPT = Path("/ws/external/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-fixed.pth")

print(f"Loading checkpoint from: {OLD_CKPT}")
ckpt = torch.load(OLD_CKPT, map_location="cpu")

# mmdet 계열은 보통 {"meta": ..., "state_dict": {...}} 구조
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

def need_convert(key: str, w: torch.Tensor) -> bool:
    """pts_middle_encoder의 5D conv weight만 골라서 변환"""
    if not key.startswith("pts_middle_encoder"):
        return False
    if not isinstance(w, torch.Tensor):
        return False
    if w.ndim != 5:
        return False  # conv weight만
    # 너가 띄운 로그 기준 out-channel 후보
    if w.shape[0] not in (16, 32, 64, 128):
        return False
    return True

converted = []
skipped = []

for k, w in list(state_dict.items()):
    if need_convert(k, w):
        old_shape = tuple(w.shape)  # [out, kx, ky, kz, in]
        # [out, kx, ky, kz, in] -> [kx, ky, kz, in, out]
        w_new = w.permute(1, 2, 3, 4, 0).contiguous()
        new_shape = tuple(w_new.shape)
        state_dict[k] = w_new
        converted.append((k, old_shape, new_shape))
    else:
        skipped.append(k)

print("\n=== Converted weights (pts_middle_encoder) ===")
for k, old_s, new_s in converted:
    print(f"{k}: {old_s} -> {new_s}")

if not converted:
    print("⚠ 변환된 weight가 없음. key prefix나 shape 조건을 다시 확인해야 함.")

# 다시 ckpt 포맷 맞춰서 저장
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    ckpt["state_dict"] = state_dict
else:
    ckpt = state_dict

torch.save(ckpt, NEW_CKPT)
print(f"\n✅ Saved fixed checkpoint to: {NEW_CKPT}")