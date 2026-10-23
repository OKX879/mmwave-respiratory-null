import json
import pickle
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(r'C:\Users\yxh\Desktop\db_records')
R_BIN_AXIS = 0.31724          # 数据自带距离轴步进
C = 3e8

print('%-6s %-18s %-5s %-6s %-6s %-6s %-8s %-8s' % (
    'sid', 'cond', 'bin', 'n50', 'n20', 'n10', 'fbeat_kHz', 'R_beat_m'))
rows = []
for i in range(1, 21):
    sid = 'P%03d' % i
    for cond in ['Sitting/Rest']:
        d = ROOT / sid / cond
        cfg = json.load(open(d / 'radar_chirpConfig.json'))
        cube = pickle.loads(zlib.decompress((d / 'radar_rFFTs.zlib').read_bytes()))[0]
        # 运动能量剖面（通道平均后逐 bin 时间 std）
        zm = cube.mean(axis=1).T                 # (bin, T)
        e = np.std(zm, axis=1)
        b = int(np.argmax(e))
        en = e / e.max()
        n50 = int((en >= 0.5).sum())
        n20 = int((en >= 0.2).sum())
        n10 = int((en >= 0.1).sum())

        # 用 IFFT 还原 beat 信号，测相位增量 -> beat 频率 -> 距离
        ch = 0
        x = np.fft.ifft(cube[:, ch, :], axis=1)     # (T, 64) 复数 beat 采样
        # 取运动最强帧附近 50 帧，逐帧相位增量中位
        idx = np.argsort(np.std(np.abs(x), axis=1))[-50:]
        dphi = np.angle(x[idx, 1:] * np.conj(x[idx, :-1]))
        fbeat = float(np.median(dphi) / (2 * np.pi) * cfg['SAMPLING_RATE'])
        R_beat = fbeat * C / (2 * cfg['SLOPE'])
        rows.append((b, n50, n20, n10, fbeat, R_beat))
        print('%-6s %-18s %-5d %-6d %-6d %-6d %-8.1f %-8.3f' % (
            sid, cond, b, n50, n20, n10, fbeat / 1e3, R_beat))

a = np.array([r[:4] for r in rows], dtype=float)
print('\nmotion-energy concentration (20 records):')
print('  bins >=50%% max: median %.0f  (range %d-%d)' % (
    np.median(a[:, 1]), a[:, 1].min(), a[:, 1].max()))
print('  bins >=20%% max: median %.0f  (range %d-%d)' % (
    np.median(a[:, 2]), a[:, 2].min(), a[:, 2].max()))
print('  bins >=10%% max: median %.0f  (range %d-%d)' % (
    np.median(a[:, 3]), a[:, 3].min(), a[:, 3].max()))

# 两种带宽假设下，0.6 m 目标应落在哪个 bin、64 bin 能覆盖多远
for B in (0.48e9, 4e9):
    rb = C / (2 * B)
    print('B=%.2f GHz -> bin %.2f cm, 0.6 m 目标在 bin %d, 64 bins 覆盖 %.2f m' % (
        B / 1e9, rb * 100, round(0.6 / rb), 64 * rb))
