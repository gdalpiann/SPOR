# scripts/05_plot_images_embedding.py
# Versão "estilo antigo": sem argparse, com constantes no topo
# Usa scatterplot (sct.circles/sct.images) + DGrid/DGridAdaptive e salva tudo organizado

import os
import pickle
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import scatterplot as sct
from dgrid import DGrid
from SPOR import DGridAdaptive

# ======== CONFIG =========
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Escolha o arquivo .pkl gerado no 03 (TSNE/UMAP/LLE/etc.)
#PICKLE_NAME = "round_inbetween_cigar_TSNE_2D.pkl"
#PICKLE_NAME = "round_inbetween_cigar_UMAP_2D.pkl"
PICKLE_NAME = "round_inbetween_cigar_LLE_2D.pkl"
#PICKLE_NAME = "round_inbetween_cigar_PCA_2D.pkl"
#PICKLE_NAME = "round_inbetween_cigar_ISO_2D.pkl"

# Tamanho das thumbnails carregadas (pixels do lado maior)
THUMB_SIZE = 96

# Tamanho (em “unidades do sct”) do glifo renderizado no canvas (para thumbnails e grids)
GLYPH_SIZE = 0.5

# Raio dos pontos (círculos) em unidades do embedding — ajuste fino conforme a escala do seu UMAP/TSNE
POINT_RADIUS = 0.03  # se ficar grande/pequeno, ajuste aqui (ex.: 0.02 ~ 0.06)

# Quais classes (pelo NOME em class_map) devem ter prioridade no DGridAdaptive
PRIORITY_NAMES = ["inbetween"]   # deixe [] se não quiser prioridade

# Tamanhos / parâmetros de figura
FIGSIZE = (10, 10)
DPI = 600
# =========================


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_thumbnails(paths_rel, thumb_size=48):
    """Lê imagens relativas a DATA_DIR, converte pra RGB, redimensiona e empilha em np.uint8 [N,H,W,3]."""
    imgs = []
    for rp in paths_rel:
        p = os.path.join(DATA_DIR, rp)
        if not os.path.exists(p):
            imgs.append(None)
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = thumb_size / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im = im.resize((new_w, new_h), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.uint8))
    return imgs


def valid_images_to_uniform_canvas(img_list):
    """
    sct.images aceita imagens HxWx3, mas é mais estável se todas tiverem o mesmo tamanho.
    Aqui, padronizamos preenchendo com borda preta até (Hmax, Wmax).
    """
    hmax = max(img.shape[0] for img in img_list)
    wmax = max(img.shape[1] for img in img_list)
    out = []
    for img in img_list:
        h, w, _ = img.shape
        canvas = np.zeros((hmax, wmax, 3), dtype=np.uint8)
        y0 = (hmax - h) // 2
        x0 = (wmax - w) // 2
        canvas[y0:y0+h, x0:x0+w, :] = img
        out.append(canvas)
    return np.stack(out, axis=0)


def save_sct_images(coords, images, outstem, glyph_size=GLYPH_SIZE):
    valid = [(xy, img) for xy, img in zip(coords, images) if img is not None]
    coords_arr = np.asarray([v[0] for v in valid], dtype=float)
    imgs_arr = valid_images_to_uniform_canvas([v[1] for v in valid])

    sct.images(coords_arr, imgs_arr,
               glyph_width=glyph_size, glyph_height=glyph_size,
               label=None, alpha=1.0, figsize=FIGSIZE, fontsize=6)
    ax = plt.gca(); ax.set_axis_off()
    png_path = f"{outstem}.png"
    pdf_path = f"{outstem}.pdf"
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"🖼️  Salvo: {png_path}")


def save_sct_points_circles(coords, class_names, outstem, point_diam=0.06):
    """
    Desenha o embedding como PONTOS usando sct.circles,
    com as mesmas cores do seu Matplotlib:
      round -> red, inbetween -> blue, cigar -> green
    point_diam controla o diâmetro do ponto nas unidades do embedding.
    """
    coords = np.asarray(coords, dtype=float)

    ordered_classes = ["round", "inbetween", "cigar"]
    color_list = ["red", "blue", "green"]
    class_to_id = {name: i for i, name in enumerate(ordered_classes)}
    labels_num = np.array([class_to_id.get(name, -1) for name in class_names], dtype=float)

    has_other = np.any(labels_num < 0)
    if has_other:
        labels_num[labels_num < 0] = 3
        color_list = color_list + ["gray"]

    cmap = mcolors.ListedColormap(color_list)

    sct.circles(
        coords,
        glyph_width=point_diam,
        glyph_height=point_diam,
        label=labels_num,
        cmap=cmap,
        alpha=0.8,
        figsize=FIGSIZE,
        linewidth=0.0,
        edgecolor='none',
    )
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    png_path = f"{outstem}.png"
    pdf_path = f"{outstem}.pdf"
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"🔵  Pontos (círculos) salvos: {png_path}")


def main():
    pk_path = os.path.join(OUT_DIR, PICKLE_NAME)
    if not os.path.exists(pk_path):
        raise FileNotFoundError(f"Pickle não encontrado: {pk_path}")

    data = load_pickle(pk_path)
    X = np.asarray(data["X_embedded"], dtype=float)    # (N,2)
    y = np.asarray(data["y"])                          # rótulos numéricos (1,2,3,...)
    class_map = data.get("class_map", None)            # dict {1:'round', ...}
    img_paths_rel = data.get("img_paths", None)        # lista de caminhos relativos

    if img_paths_rel is None:
        raise KeyError("Este pickle não contém 'img_paths'. Gere novamente com o 03_compute_embeddings.py atualizado.")

    # nomes de classe por amostra
    if class_map is not None:
        inv_map = {int(k): v for k, v in class_map.items()}
        cls_names = np.array([inv_map.get(int(cid), "unknown") for cid in y], dtype=object)
    else:
        cls_names = np.array(["unknown"] * len(X), dtype=object)

    # carregar thumbnails
    thumbs = load_thumbnails(img_paths_rel, thumb_size=THUMB_SIZE)

    # filtra amostras com imagem ausente
    valid_mask = np.array([im is not None for im in thumbs], dtype=bool)
    if not valid_mask.all():
        print(f"⚠️  {np.count_nonzero(~valid_mask)} imagens faltantes serão ignoradas.")
    Xv = X[valid_mask]
    thumbs_v = [im for (im, ok) in zip(thumbs, valid_mask) if ok]
    cls_names_v = cls_names[valid_mask]

    # out dir
    stem = os.path.splitext(PICKLE_NAME)[0]
    out_dir = os.path.join(OUT_DIR, "grids", stem)
    os.makedirs(out_dir, exist_ok=True)

    # (A0) Embedding como PONTOS (círculos)
    print("A0) Embedding (pontos/círculos via sct.circles)...")
    save_sct_points_circles(Xv, cls_names_v, os.path.join(out_dir, "a_points"),
                            point_diam=2 * POINT_RADIUS)  # radius -> diameter

    # (A) Embedding "cru" com thumbnails
    print("A) Embedding original (thumbnails no 2D)...")
    save_sct_images(Xv, thumbs_v, os.path.join(out_dir, "a_embedding"), glyph_size=GLYPH_SIZE)

    # (B) DGrid delta=4 e 8 (thumbnails) — mantém do seu pipeline
    for delta in (4, 8):
        print(f"B) DGrid delta={delta} ...")
        dgrid = DGrid(glyph_width=GLYPH_SIZE, glyph_height=GLYPH_SIZE, delta=delta)
        coords = dgrid.fit_transform(Xv)
        save_sct_images(coords, thumbs_v, os.path.join(out_dir, f"b_dgrid_delta{delta}"), glyph_size=GLYPH_SIZE)

    # (B-points) DGrid delta=1 como PONTOS
    print("B-points) DGrid delta=1 (pontos/círculos)...")
    dgrid_pts = DGrid(glyph_width=2 * POINT_RADIUS, glyph_height=2 * POINT_RADIUS, delta=1)
    coords_pts = dgrid_pts.fit_transform(Xv)
    save_sct_points_circles(coords_pts, cls_names_v,
                            os.path.join(out_dir, "b_dgrid_delta1_points"),
                            point_diam=2 * POINT_RADIUS)

    # (C) DGridAdaptive (GridFair) sem prioridade, ws = 1.0, 0.9, 0.8 (thumbnails)
    for ws in (1.0, 0.9, 0.8):
        print(f"C) DGridAdaptive (gridfair, ws={ws}, sem prioridade) ...")
        adapt = DGridAdaptive(glyph_size=GLYPH_SIZE, white_space_ratio=ws, sampling_strategy="gridfair")
        coords = adapt.fit_transform(Xv, labels=None)
        idx_real = [p["id"] for p in adapt.grid if not p["dummy"]]
        thumbs_sel = [thumbs_v[i] for i in idx_real]
        coords_sel = coords
        save_sct_images(coords_sel, thumbs_sel,
                        os.path.join(out_dir, f"c_adaptive_ws{int(ws*100)}_nopriority"),
                        glyph_size=GLYPH_SIZE)

    # (C-points) DGridAdaptive ws="auto_overlap_free" como PONTOS
    # OBS: seu DGridAdaptive deve aceitar a string "auto_overlap_free".
    # Se não aceitar, troque por um valor numérico (ex.: 1.0) ou adapte a lib.
    print('C-points) DGridAdaptive (gridfair, ws="auto_overlap_free", pontos)...')
    adapt_auto = DGridAdaptive(glyph_size=2 * POINT_RADIUS,
                               white_space_ratio="auto_overlap_free",
                               sampling_strategy="gridfair")
    coords_auto = adapt_auto.fit_transform(Xv, labels=None)
    idx_real_auto = [p["id"] for p in adapt_auto.grid if not p["dummy"]]
    cls_sel_auto = cls_names_v[idx_real_auto]
    save_sct_points_circles(coords_auto, cls_sel_auto,
                            os.path.join(out_dir, "c_adaptive_wsauto_points"),
                            point_diam=2 * POINT_RADIUS)

    # (D) DGridAdaptive com prioridade (thumbnails), mantendo seu pipeline
    if PRIORITY_NAMES:
        pr_mask = np.isin(cls_names_v, PRIORITY_NAMES).astype(int)
        for ws in (1.0, 0.9, 0.8):
            print(f"D) DGridAdaptive (gridfair, ws={ws}, prioridade={PRIORITY_NAMES}) ...")
            adapt_p = DGridAdaptive(glyph_size=GLYPH_SIZE,
                                    white_space_ratio=ws,
                                    sampling_strategy="gridfair",
                                    priority_labels=[1])
            coords_p = adapt_p.fit_transform(Xv, labels=pr_mask)
            idx_real_p = [p["id"] for p in adapt_p.grid if not p["dummy"]]
            thumbs_sel_p = [thumbs_v[i] for i in idx_real_p]
            coords_sel_p = coords_p
            tag = "_".join(PRIORITY_NAMES).replace(" ", "-")
            save_sct_images(coords_sel_p, thumbs_sel_p,
                            os.path.join(out_dir, f"d_adaptive_ws{int(ws*100)}_priority_{tag}"),
                            glyph_size=GLYPH_SIZE)

    print(f"✅ Tudo salvo em: {out_dir}")


if __name__ == "__main__":
    main()
