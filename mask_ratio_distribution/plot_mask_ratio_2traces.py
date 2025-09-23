import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

adetailor_mask_file = "./adetailor_mask_ratio.txt"
with open(adetailor_mask_file, "r") as f:
    line = f.readline().strip()
adetailor_mask = np.array(eval(line))

spring_festival_mask = np.load("./spring_festival_mask_ratio_original.npy")
inpaint_mask = np.load("./controlnet_inpaint_mask_ratios_1918.npy")

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
fontsize = 32

# plot the distribution of spring_festival_mask
num_bins = 100
bins_count = [0] * num_bins
for mask_ratio in spring_festival_mask:
    if mask_ratio < 0.01:
        bins_count[0] += 1
    elif mask_ratio < 0.02:
        bins_count[1] += 1
    else:
        bins_count[int(mask_ratio * 100)] += 1
# plot the distribution of spring_festival_mask
ax[0].bar(range(num_bins), np.array(bins_count) / len(spring_festival_mask), width=1)
ax[0].set_xlabel('Mask Ratio', fontsize=fontsize)
ax[0].set_ylabel('Density', fontsize=fontsize)
ax[0].text(0.5, 0.9, 'mean: {:.2f}'.format(np.mean(spring_festival_mask)), fontsize=fontsize, ha='left', va='center', transform=ax[0].transAxes)
# ax[0].text(0.5, 0.75, 'std: {:.2f}'.format(np.std(spring_festival_mask)), fontsize=fontsize, ha='left', va='center', transform=ax[0].transAxes)
# ax[0].set_title('Spring Festival Mask', fontsize=fontsize)

katz_mask_ratio = np.concatenate([inpaint_mask, adetailor_mask])
# plot the distribution of inpaint_mask
bins_count = [0] * num_bins
for mask_ratio in katz_mask_ratio:
    if mask_ratio < 0.01:
        bins_count[0] += 1
    elif mask_ratio < 0.02:
        bins_count[1] += 1
    else:
        bins_count[int(mask_ratio * 100)] += 1

ax[1].bar(range(num_bins), np.array(bins_count) / len(katz_mask_ratio), width=1)
ax[1].set_xlabel('Mask Ratio', fontsize=fontsize)
ax[1].text(0.5, 0.9, 'mean: {:.2f}'.format(np.mean(katz_mask_ratio)), fontsize=fontsize, ha='left', va='center', transform=ax[1].transAxes)
# ax[1].text(0.5, 0.75, 'std: {:.2f}'.format(np.std(katz_mask_ratio)), fontsize=fontsize, ha='left', va='center', transform=ax[1].transAxes)

# ax[1].set_ylabel('Density', fontsize=fontsize)
# ax[1].set_title('Inpaint Mask', fontsize=fontsize)



ax[0].grid(axis='y')
ax[1].grid(axis='y')

ax[0].tick_params(axis='y', labelsize=fontsize)
ax[0].tick_params(axis='x', labelsize=fontsize)
ax[1].tick_params(axis='y', labelsize=fontsize)
ax[1].tick_params(axis='x', labelsize=fontsize)

ax[0].set_yticks([0.0, 0.05, 0.1])
ax[1].set_yticks([0.0, 0.05, 0.1, 0.15])

ax[0].set_xticks(range(0, 101, 20))
ax[1].set_xticks(range(0, 101, 20))

ax[0].set_xticklabels(np.arange(0, 101, 20)/100.0)
ax[1].set_xticklabels(np.arange(0, 101, 20)/100.0)

fig.tight_layout()
plt.savefig("mask_ratio_2traces.pdf", format='pdf', bbox_inches='tight', pad_inches=0.03)
