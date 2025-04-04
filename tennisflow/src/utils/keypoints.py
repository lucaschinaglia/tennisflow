def denormalize_keypoints(normalized_keypoints, width, height):
    """Denormalize keypoints back to pixel coordinates."""
    # Ensure we work on a copy
    denormalized_kps = normalized_keypoints.copy() * np.array([width, height])
    return denormalized_kps.astype(int) 