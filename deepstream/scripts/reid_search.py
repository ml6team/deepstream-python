import glob
import os
import json
import logging
import time
import shutil

import numpy as np
import cv2

OUTPUT_DIR = "~/deepstream-python/output"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
REID_FEATURES_PATH = os.path.join(OUTPUT_DIR, "reid_features.json")
GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class NumpySearch:

    def __init__(self, metric: str = 'cosine', top_n: int = 5):
        self.metric = metric
        self.top_n = top_n

    def _compute_dist_mat(self, query_embeddings: np.array,
                          gallery_embeddings: np.array) -> np.array:
        start = time.time()
        x = query_embeddings
        y = gallery_embeddings

        if self.metric == 'cosine':
            x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
            y_norm = np.linalg.norm(y, axis=-1, keepdims=True)

            x = x / x_norm
            y = y / y_norm

        dist_mat = 1 - np.matmul(x, np.transpose(y))
        logging.info(f"Computed {dist_mat.shape} distance matrix in {time.time() - start}s.")

        return dist_mat

    def search(self, q_cam_ids: np.array, q_p_ids: np.array, q_features: np.array,
               g_cam_ids: np.array, g_p_ids: np.array, g_features: np.array):
        dist_mat = self._compute_dist_mat(q_features, g_features)

        num_gallery = len(g_features)
        if self.top_n > num_gallery:
            top_n = num_gallery
        else:
            top_n = self.top_n

        top_camera_ids = []
        top_identity_ids = []
        top_distances = []
        for i, (q_cam_id, q_p_id) in enumerate(zip(q_cam_ids, q_p_ids)):
            keep = (g_p_ids != q_p_id) | (g_cam_ids != q_cam_id)
            distances = np.array([dist_mat[i, j] for j in range(num_gallery)])[keep]
            if top_n == num_gallery:
                top_ids = np.argsort(distances)
            else:
                idx = np.argpartition(distances, top_n)[:top_n]
                top_ids = idx[np.argsort(distances[idx])]
                distances = distances[idx]

            # for top_id in top_ids:
            top_camera_ids.append(g_cam_ids[keep][top_ids].tolist())
            top_identity_ids.append(g_p_ids[keep][top_ids].tolist())
            top_distances.append(distances.tolist())

        return top_camera_ids, top_identity_ids, top_distances, dist_mat

    def visualize_ranked_results(
            self, distmat, dataset, data_type, width=128, height=256, save_dir=''
    ):
        """Visualizes ranked results.
        Supports both image-reid and video-reid.
        For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
        saved in folders each containing a tracklet.
        Args:
            distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
            dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
                tuples of (img_path(s), pid, camid, dsetid).
            data_type (str): "image" or "video".
            width (int, optional): resized image width. Default is 128.
            height (int, optional): resized image height. Default is 256.
            save_dir (str): directory to save output images.
        """
        num_q, num_g = distmat.shape
        os.makedirs(save_dir, exist_ok=True)

        logging.info('# query: {}\n# gallery {}'.format(num_q, num_g))
        logging.info('Visualizing top-{} ranks ...'.format(self.top_n))

        query, gallery = dataset
        if not num_q == len(query): raise ValueError
        if not num_g == len(gallery): raise ValueError

        indices = np.argsort(distmat, axis=1)

        def _cp_img_to(src, dst, rank, prefix, matched=False):
            """
            Args:
                src: image path or tuple (for vidreid)
                dst: target directory
                rank: int, denoting ranked position, starting from 1
                prefix: string
                matched: bool
            """
            if isinstance(src, (tuple, list)):
                if prefix == 'gallery':
                    suffix = 'TRUE' if matched else 'FALSE'
                    dst = os.path.join(
                        dst, prefix + '_top' + str(rank).zfill(3)
                    ) + '_' + suffix
                else:
                    dst = os.path.join(dst, prefix + '_top' + str(rank).zfill(3))
                os.makedirs(dst, exist_ok=True)
                for img_path in src:
                    shutil.copy(img_path, dst)
            else:
                dst = os.path.join(
                    dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                         os.path.basename(src)
                )
                shutil.copy(src, dst)

        for q_idx in range(num_q):
            qimg_path, qpid, qcamid = query[q_idx][:3]
            qimg_path_name = qimg_path[0] if isinstance(
                qimg_path, (tuple, list)
            ) else qimg_path

            if data_type == 'image':
                qimg = cv2.imread(qimg_path)
                qimg = cv2.resize(qimg, (width, height))
                qimg = cv2.copyMakeBorder(
                    qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                # resize twice to ensure that the border width is consistent across images
                qimg = cv2.resize(qimg, (width, height))
                num_cols = self.top_n + 1
                grid_img = 255 * np.ones(
                    (height, num_cols * width + self.top_n * GRID_SPACING + QUERY_EXTRA_SPACING, 3),
                    dtype=np.uint8
                )
                grid_img[:, :width, :] = qimg
            else:
                qdir = os.path.join(
                    save_dir, os.path.basename(os.path.splitext(qimg_path_name)[0])
                )
                os.makedirs(qdir, exist_ok=True)
                _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

            rank_idx = 1
            for g_idx in indices[q_idx, :]:
                gimg_path, gpid, gcamid = gallery[g_idx][:3]
                invalid = (qpid == gpid) & (qcamid == gcamid)

                if not invalid:
                    matched = gpid == qpid
                    if data_type == 'image':
                        border_color = GREEN if matched else RED
                        gimg = cv2.imread(gimg_path)
                        gimg = cv2.resize(gimg, (width, height))
                        gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT,
                                                  value=border_color)
                        gimg = cv2.resize(gimg, (width, height))
                        start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        end = (rank_idx + 1) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        grid_img[:, start:end, :] = gimg
                    else:
                        _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery',
                                   matched=matched)

                    rank_idx += 1
                    if rank_idx > self.top_n:
                        break

            if data_type == 'image':
                imname = os.path.basename(os.path.splitext(qimg_path_name)[0])
                cv2.imwrite(os.path.join(save_dir, imname + '.jpg'), grid_img)

            if (q_idx + 1) % 100 == 0:
                print('- done {}/{}'.format(q_idx + 1, num_q))

        logging.info('Done. Images have been saved to "{}" ...'.format(save_dir))


def main(q_id: str, q_camid: str, num_embeddings: int = 2):
    with open(REID_FEATURES_PATH, "r") as json_file:
        reid_features = json.load(json_file)

    q_features = np.array(reid_features[q_id])[:num_embeddings]
    q_camids = np.repeat(q_camid, len(q_features))
    q_pids = np.repeat(q_id, len(q_features))
    q_path = glob.glob(os.path.join(CROPS_DIR, f"src_{q_camid}", f"obj_{q_id}_cls_0", "*.jpg"))[0]
    q_paths = np.repeat(q_path, len(q_features))
    print(q_pids.shape)
    print(q_camids.shape)
    print(q_features.shape)
    query = [(x, y, z) for x, y, z in zip(q_paths, q_pids, q_camids)]

    g_pids_list = []
    g_features_list = []
    g_paths_list = []
    for track_id, track in reid_features.items():
        g_path = glob.glob(os.path.join(CROPS_DIR,
                                        f"src_{q_camid}", f"obj_{track_id}_cls_0", "*.jpg"))[0]
        for obj in track:
            g_pids_list.append(str(track_id))
            g_features_list.append(obj)
            g_paths_list.append(g_path)

    g_pids = np.array(g_pids_list)
    g_camids = np.repeat("0", len(g_pids))
    g_features = np.array(g_features_list)
    g_paths = np.array(g_paths_list)
    print(g_pids.shape)
    print(g_camids.shape)
    print(g_features.shape)
    gallery = [(x, y, z) for x, y, z in zip(g_paths, g_pids, g_camids)]

    search_model = NumpySearch()
    top_camera_ids, top_identity_ids, top_distances, dist_mat = search_model.search(
        q_camids, q_pids, q_features, g_camids, g_pids, g_features
    )
    print(top_camera_ids)
    print(top_identity_ids)
    print(top_distances)

    search_model.visualize_ranked_results(dist_mat, (query, gallery),
                                          data_type="image", save_dir=OUTPUT_DIR)


if __name__ == '__main__':
    main("4", "0")
