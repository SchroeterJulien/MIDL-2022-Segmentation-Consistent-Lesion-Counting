import torch
from scipy import ndimage
import numpy as np

cca_structure = ndimage.generate_binary_structure(3, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def CountFunction(segmentation_map, max_occurence=5, threshold=0.1, max_instance=20, voxel_threshold=0):
    """
    Function mapping lesion segmentation maps (sigmoid) to lesion count distributions.
    :param segmentation_map: segmentation map [torch tensor] of shape (batch, D, H, W)
    :param max_occurence: extent of the count distribution (e.g., 5 gives, 0,1,2,3,4+ as count classes) [int]
    :param threshold: threshold for the binarization of the segmentation map ($tau$ in paper) [float]
    :param max_instance: maximum number of lesions [int]
    :param voxel_threshold: size-based threshold to filter region proposal [int]
    :return: Lesion Count distribution [torch tensor] of shape (batch, max_occurence)
    """

    # 1. Lesion Candidates Identification (Section 3.1.1)

    with torch.no_grad():
        preds_copy = segmentation_map.detach().clone()
        preds_copy.requires_grad = False
        preds_copy = (preds_copy > threshold).double()

    # mask shape: (batch, instances, D, H, W)
    all_masks = torch.zeros(size=(segmentation_map.size(0), max_instance, segmentation_map.size(1),
                                  segmentation_map.size(2), segmentation_map.size(3)), device=device,
                            requires_grad=False)
    cc_counts = torch.zeros(size=(segmentation_map.size(0), 1), device=device)
    for i, pred in enumerate(preds_copy):
        with torch.no_grad():
            # use connected components to get regions
            cc_output, cc_output_num_features = ndimage.label(pred.squeeze(0).detach().cpu().numpy(),
                                                              structure=cca_structure)
            cc_counts[i] = cc_output_num_features

            # (optional) Select Lesion based on size
            segment_idx, segment_count = np.unique(cc_output, return_counts=True)
            segment_idx = segment_idx[segment_count > voxel_threshold]
            count_sort_ind = np.argsort(-segment_count[segment_count > voxel_threshold])  # Sort by size
            unique_segment_idx = segment_idx[count_sort_ind][:max_instance + 1]

            cc_output = torch.from_numpy(cc_output).to(device)

        jj = 0
        for seg_idx in unique_segment_idx:
            # Ignore background
            if seg_idx > 0:
                # use torch.no_grad to be sure no gradients are added to the computational graph
                with torch.no_grad():
                    mask = (cc_output == seg_idx).double()

                # gradient should be tracked here
                all_masks[i, jj, :, :, :] = mask
                jj += 1

    # 2. Lesion Existence Probability (Section 3.1.2)
    # lesion_existence_prob = torch.amax(all_masks[:, :max(jj, 1)] * segmentation_map.unsqueeze(1), dim=(2, 3, 4))
    # long version that work on all tested version of pytorch
    lesion_existence_prob = \
        torch.max(torch.max(torch.max(all_masks[:, :max(jj, 1)]
                                      * segmentation_map.unsqueeze(1), dim=2)[0], dim=2)[0], dim=2)[0]

    # 3. Poisson-binomial Counting (Section 3.1.3)
    count_predictions = PoissonBinomialCounting(lesion_existence_prob, max_occurence=max_occurence).to(device)

    return count_predictions, cc_counts


def PoissonBinomialCounting(instance_probability, max_occurence=5):
    """
    Computes the Poisson-binomial distribution
    # credit: https://github.com/SchroeterJulien/ACCV-2020-Subpixel-Point-Localization
    :param instance_probability: individual instance probabilities that have to be aggregated to a count distribution.
                                 [torch tensor] of shape: (batch_size, n_instances)
    :param max_occurence: extent of the count distribution (e.g., 5 gives, 0,1,2,3,4+ as count classes) [int]
    :return: Poisson-binomial count distribution  [torch tensor] of shape: (batch_size, max_occurence)
    """
    contribution = torch.unbind(instance_probability, 1)

    count_prediction = torch.zeros(instance_probability.size()[0], max_occurence, device=device)
    count_prediction[:, 0] = 1
    # Recursion from: http://proceedings.mlr.press/v97/schroeter19a.html
    for increment in contribution:
        mass_movement = (count_prediction * increment.unsqueeze(1))[:, :max_occurence - 1]
        move = - torch.cat([mass_movement, torch.zeros(count_prediction.size()[0], 1, device=device)], axis=1) \
               + torch.cat([torch.zeros(count_prediction.size()[0], 1, device=device), mass_movement], axis=1)

        count_prediction = count_prediction + move

    return count_prediction
