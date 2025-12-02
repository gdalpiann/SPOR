import math
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import time
from typing import Optional, List, Literal, Union
from numba import njit

RETURN_TYPE = Literal['coord', 'index']

@njit(parallel=False, nogil=True)
def density_calculation_numba(count_map, mask, mask_size, cell_rows, cell_cols, nr_rows, nr_columns):
    densities = np.zeros(cell_rows.shape[0], dtype=np.float32)
    for idx in range(cell_rows.shape[0]):
        row = cell_rows[idx]
        col = cell_cols[idx]
        density = 0.0
        for i in range(mask_size):
            for j in range(mask_size):
                r = row - (mask_size // 2) + j
                c = col - (mask_size // 2) + i
                if (0 <= r < nr_rows) and (0 <= c < nr_columns):
                    density += mask[i, j] * count_map[r, c]
        densities[idx] = density
    return densities


class DGridAdaptive:
    def __init__(self,
                 glyph_size: float = 1.0,
                 white_space_ratio: Union[float, str] = 1.0,
                 return_type: RETURN_TYPE = 'coord',
                 priority_labels: Optional[List[int]] = None,
                 random_state: Optional[int] = None,
                 sampling_strategy: str = 'gridfair',   # <— NOVO
                 type_search: int = 2                   # <— NOVO (2=medoide, 1=primeiro, 0=random)
                 ):
        self.glyph_size = glyph_size
        self.white_space_ratio = white_space_ratio
        self.return_type = return_type
        self.priority_labels = priority_labels or []
        self.random_state = random_state
        self.grid = []
        self.sampling_strategy = sampling_strategy.lower()
        self.type_search = type_search

    def fit_transform(self, y: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        
        # Verifica se white_space_ratio é maior que 1
        if isinstance(self.white_space_ratio, (float, int)) and self.white_space_ratio > 1.0:
            raise ValueError(f"O valor de white_space_ratio ({self.white_space_ratio}) não pode ser maior que 1.")
        if isinstance(self.white_space_ratio, str) and self.white_space_ratio != "auto_overlap_free":
            raise ValueError(f"white_space_ratio string must be 'auto_overlap_free', got '{self.white_space_ratio}'.")
        
        total_start = time.time()
        print("▶️ Iniciando DGridAdaptive...")

        # 1. Bounding box e grid
        start = time.time()
        x_min, y_min = np.min(y, axis=0)
        x_max, y_max = np.max(y, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        nr_columns = math.ceil(width / self.glyph_size)
        nr_rows = math.ceil(height / self.glyph_size)
        n_cells = nr_rows * nr_columns
        print(f"⏱️ Bounding box e grid: {time.time() - start:.2f} s")

        n_points = y.shape[0]  # número de pontos reais

        # 2. Atribuição de pontos reais às células
        start = time.time()
        grid_map = dict()
        points_real = []
        for idx, (x0, y0) in enumerate(y):
            col = min(int((x0 - x_min) / self.glyph_size), nr_columns - 1)
            row = min(int((y0 - y_min) / self.glyph_size), nr_rows - 1)
            label = labels[idx] if labels is not None else None
            points_real.append({'id': idx, 'x': x0, 'y': y0, 'row': row, 'col': col, 'dummy': False, 'label': label})
            grid_map.setdefault((row, col), []).append(idx)
        print(f"⏱️ Mapeamento dos pontos reais: {time.time() - start:.2f} s")

        #mudança filtro vizinhos dummy points
        self._grid_map = grid_map

        # 3. Dummy points candidates: todas células vazias
        start = time.time()
        cells_vazias = [(row, col) for row in range(nr_rows) for col in range(nr_columns)
                        if (row, col) not in grid_map]
        dummy_points_candidates = self._get_dummy_points_candidates(
            cells_vazias, grid_map, points_real, x_min, x_max, y_min, y_max, nr_columns, nr_rows
        )
        print(f"⏱️ Geração dos dummy points candidates: {time.time() - start:.2f} s")

        n_dummy_total = len(dummy_points_candidates)
        min_ws = 0.0  # valor padrão

        # CALCULAR O WHITE_SPACE_RATIO MÍNIMO NECESSÁRIO
        if n_points >= n_cells:
            min_ws = 0.0
        else:
            n_dummy_final = n_cells - n_points
            min_ws = n_dummy_final / n_dummy_total if n_dummy_total > 0 else 0.0

        # Corrige white_space_ratio se 'auto_overlap_free'
        ws_input = self.white_space_ratio
        if isinstance(ws_input, str):
            # Correção automática para overlap free
            if n_points > n_cells:
                print(f"[AVISO] Não há células suficientes ({n_cells}) para mostrar todos os pontos ({n_points}). Ajustando white_space_ratio para 0.0 (mínimo possível).")
                self.white_space_ratio = 0.0
            else:
                print(f"[INFO] Ajustando white_space_ratio automaticamente para garantir overlap-free: ws = {min_ws:.4f}")
                self.white_space_ratio = min_ws

        # Corrige white_space_ratio se menor que o mínimo necessário
        if isinstance(self.white_space_ratio, (float, int)) and self.white_space_ratio < min_ws:
            print(f"[AVISO] O valor de white_space_ratio fornecido ({self.white_space_ratio:.4f}) é menor do que o mínimo necessário ({min_ws:.4f}) para acomodar todos os pontos sem bugs. Ajustando para {min_ws:.4f}.")
            self.white_space_ratio = min_ws

        # 4. Seleção dos dummy points finais (whitespace controlado + espalhamento)
        start = time.time()
        n_dummy_final = int(round(self.white_space_ratio * n_dummy_total))
        selected_dummy_points = self._select_dummy_points(dummy_points_candidates, n_dummy_final)
        print(f"⏱️ Seleção dos dummy points finais: {time.time() - start:.2f} s")

        # 5. Determinação do número máximo de pontos reais a serem exibidos
        n_real_max = n_cells - n_dummy_final

        # 6. Filtragem dos pontos reais (com ou sem prioridade de classe)
        start = time.time()
        filtered_points_real = self._filter_real_points(points_real, n_real_max)
        print(f"⏱️ Filtragem dos pontos reais: {time.time() - start:.2f} s")
        print(f"   → Número de pontos reais finais: {len(filtered_points_real)}")

        # 7. Assignment global recursivo (DGrid)
        start = time.time()
        assignment_list = filtered_points_real + selected_dummy_points
        assigned = self._assignment_global(assignment_list, nr_rows, nr_columns)
        print(f"⏱️ Assignment global recursivo: {time.time() - start:.2f} s")

        # 8. Prepara saída
        start = time.time()
        out = []
        for p in assigned:
            if not p['dummy']:
                x = x_min + (p['col'] + 0.5) * self.glyph_size
                y = y_min + (p['row'] + 0.5) * self.glyph_size
                out.append([x, y])
        print(f"⏱️ Preparação da saída: {time.time() - start:.2f} s")
        print(f"✅ DGridAdaptive finalizado em {time.time() - total_start:.2f} s\n")
        
        self.grid = assigned  # Isso já deixa compatível!
        return np.array(out)


    def _get_dummy_points_candidates(self, cells_vazias, grid_map, points_real, x_min, x_max, y_min, y_max, nr_columns, nr_rows):
        import time
        start = time.time()

        count_map = np.zeros((nr_rows, nr_columns), dtype=np.uint32)
        for pt in points_real:
            count_map[pt['row'], pt['col']] += 1

        mask_size = int(max(3, ((x_max - x_min)*(y_max - y_min)) / (len(points_real)*self.glyph_size**2)))
        mask_size += 1 if mask_size % 2 == 0 else 0
        mask = self._gaussian_mask(mask_size, (mask_size - 1)/6.0)

        cell_rows = np.array([row for row, col in cells_vazias], dtype=np.int32)
        cell_cols = np.array([col for row, col in cells_vazias], dtype=np.int32)

        densities = density_calculation_numba(count_map, mask, mask_size, cell_rows, cell_cols, nr_rows, nr_columns)

        pts = np.array([[pt['x'], pt['y']] for pt in points_real])
        if len(pts) > 0:
            tree = KDTree(pts)
            dists = tree.query(np.array([
                [x_min + (col + 0.5)*self.glyph_size, y_min + (row + 0.5)*self.glyph_size]
                for row, col in zip(cell_rows, cell_cols)
            ]), 1)[0][:, 0]
        else:
            dists = np.zeros(len(cells_vazias))

        dummy_points_candidates = [{
            'id': -1,
            'x': x_min + (col + 0.5)*self.glyph_size,
            'y': y_min + (row + 0.5)*self.glyph_size,
            'row': row,
            'col': col,
            'dummy': True,
            'density': densities[k],
            'dist': dists[k]
        } for k, (row, col) in enumerate(zip(cell_rows, cell_cols))]

        # Ordenação exatamente como ORIGINAL (densidade crescente, distância decrescente)
        #dummy_points_candidates.sort(key=lambda d: (d['density'], -d['dist']))
        # --- preferência por distância intermediária, mantendo densidade como 1º critério ---

        dist_arr = np.array([d['dist'] for d in dummy_points_candidates], dtype=float)

        # normalização robusta 0..1 (reduz efeito de outliers)
        p10, p90 = np.percentile(dist_arr, [10, 90]) if dist_arr.size else (0.0, 1.0)
        den = (p90 - p10) if (p90 > p10) else (dist_arr.max() - dist_arr.min() + 1e-12)
        dist_n = np.clip((dist_arr - p10) / (den + 1e-12), 0.0, 1.0)

        # alvo no meio
        mu = np.median(dist_n) if dist_n.size else 0.5

        # largura da faixa intermediária (σ = 0.5 * IQR)
        q25, q75 = np.percentile(dist_n, [25, 75]) if dist_n.size else (0.25, 0.75)
        iqr = max(q75 - q25, 1e-3)
        sigma = 0.5 * iqr

        # "goodness" máximo no meio
        good = np.exp(-0.5 * ((dist_n - mu) / (sigma + 1e-12))**2)

        # anexa e ordena (crescente). como você pega o final e depois faz reversed,
        # os mais densos e com maior 'good' entram primeiro.
        for k, d in enumerate(dummy_points_candidates):
            d['_good'] = float(good[k])

        dummy_points_candidates.sort(key=lambda d: (d['density'], d['_good']))

        print(f"...Dummy points: {len(dummy_points_candidates)} candidatos gerados em {time.time()-start:.2f}s")
        return dummy_points_candidates


    def _select_dummy_points(self, dummy_points_candidates, n_dummy_final):
        if n_dummy_final == 0:
            return []

        # Mesma lógica original
        
        return dummy_points_candidates[-n_dummy_final:]

    def _run_sampling(self, points_subset, k):
        """
        Aplica a estratégia de amostragem configurada sobre 'points_subset'
        para selecionar exatamente 'k' pontos (quando possível).
        """
        if k <= 0:
            return []
        if k >= len(points_subset):
            return points_subset

        strat = (self.sampling_strategy or 'gridfair').lower()
        if strat in ('gridfair', 'fair', 'gf'):
            return self._sample_points_gridfair(points_subset, k, type_search=self.type_search)
        elif strat in ('gridfair_density', 'density', 'gfd', 'proportional'):
            return self._sample_points_gridfair_density(points_subset, k, type_search=self.type_search)
        else:
            raise ValueError(
                f"sampling_strategy inválida: {self.sampling_strategy}. "
                f"Use 'gridfair' ou 'gridfair_density'."
            )

    def _filter_real_points(self, points_real, n_real_max):
        """
        Seleciona exatamente n_real_max pontos reais.
        - Se não houver prioridade de classe -> aplica a estratégia normal.
        - Se houver prioridade: tenta incluir todos os pontos das classes prioritárias
        (até n_real_max). Se exceder, amostra somente dentro dos prioritários.
        Se sobrar capacidade, completa com a estratégia normal nos demais.
        """
        # Capacidade suficiente? nada a fazer
        if n_real_max >= len(points_real):
            return points_real

        # Se não foi passado labels, todos terão 'label' None
        has_labels = any(p.get('label') is not None for p in points_real)
        pri_list = [*self.priority_labels] if self.priority_labels else []

        # Sem prioridade configurada ou sem labels -> segue o fluxo padrão
        if not pri_list or not has_labels:
            return self._run_sampling(points_real, n_real_max)

        # Separa prioritários e não-prioritários
        priority_points = [p for p in points_real if p.get('label') in pri_list]
        other_points    = [p for p in points_real if p.get('label') not in pri_list]

        n_pri = len(priority_points)

        # Caso 1: só prioritários já excedem a capacidade
        if n_pri >= n_real_max:
            if n_pri > n_real_max:
                print(
                    f"[PRIORIDADE] Existem {n_pri} pontos de classe prioritária, "
                    f"mas a capacidade é {n_real_max}. Selecionando somente {n_real_max} "
                    f"prioritários. Para manter todos, reduza o white_space_ratio."
                )
            # Amostra apenas entre os prioritários para caber em n_real_max
            return self._run_sampling(priority_points, n_real_max)

        # Caso 2: dá para manter todos os prioritários e ainda sobra capacidade
        selected = list(priority_points)  # mantém todos os prioritários
        remaining = n_real_max - n_pri

        if remaining > 0:
            fill = self._run_sampling(other_points, remaining)
            selected.extend(fill)

            # Aviso opcional se não foi possível completar (dataset pequeno, etc.)
            if len(fill) < remaining:
                print(
                    f"[PRIORIDADE] Só foi possível selecionar {len(fill)} pontos não-prioritários "
                    f"para completar {remaining} vagas restantes."
                )

        return selected


    def _sample_points_gridfair_density(
        self,
        points_real,
        n_real_max,
        type_search=2,  # 2=medoid/local, 1=primeiro, 0=random
    ):
        import time
        t_start = time.time()
        print("\n[GridFair_density] Iniciando filtragem proporcional à densidade por grid")
        glyph_size = self.glyph_size

        if n_real_max <= 0 or len(points_real) == 0:
            print("[GridFair_density] Capacidade zero ou conjunto vazio; retornando 0 pontos.")
            return []

        coords = np.array([[p['x'], p['y']] for p in points_real])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        width, height = x_max - x_min, y_max - y_min
        n_cols = math.ceil(width / glyph_size)
        n_rows = math.ceil(height / glyph_size)
        print(f"[GridFair_density] Grid: {n_rows} rows x {n_cols} cols")

        # 1. Agrupa pontos por célula
        cell_map = {}
        for idx, (x, y) in enumerate(coords):
            col = min(int((x - x_min) / glyph_size), n_cols - 1)
            row = min(int((y - y_min) / glyph_size), n_rows - 1)
            cell_map.setdefault((row, col), []).append(idx)
        n_occ = len(cell_map)
        print(f"[GridFair_density] Células não-vazias: {n_occ}")

        # Contagens por célula e ordenação por densidade (células mais cheias primeiro)
        n_pts_in_cells = {cell: len(ixs) for cell, ixs in cell_map.items()}
        sorted_cells = sorted(cell_map.keys(), key=lambda c: n_pts_in_cells[c], reverse=True)

        cell_take = {}

        if n_real_max < n_occ:
            # Capacidade insuficiente para dar 1 por célula:
            # escolhe as n_real_max células mais densas e aloca 1 em cada
            print(f"[GridFair_density] Capacidade ({n_real_max}) < células ocupadas ({n_occ}). "
                f"Selecionando {n_real_max} células mais densas com 1 ponto cada.")
            selected_cells = sorted_cells[:n_real_max]
            for c in selected_cells:
                cell_take[c] = 1
            remaining = 0  # já atingimos n_real_max
        else:
            # Capacidade suficiente: 1 por célula + distribuição proporcional do restante
            for c in sorted_cells:
                cell_take[c] = 1
            already_selected = n_occ
            remaining = n_real_max - n_occ
            total_excess = sum(n_pts_in_cells[c] - 1 for c in sorted_cells)
            print(f"[GridFair_density] Alocado inicial: {already_selected} (1 por célula), faltam {remaining}")

            if remaining > 0 and total_excess > 0:
                # fração ideal de extras por célula
                n_extra_ideal = {
                    c: ((n_pts_in_cells[c] - 1) / total_excess) * remaining if total_excess > 0 else 0.0
                    for c in sorted_cells
                }
                # parte inteira
                for c in sorted_cells:
                    take_int = int(np.floor(n_extra_ideal[c]))
                    # não excede disponibilidade local
                    cell_take[c] = min(cell_take[c] + take_int, n_pts_in_cells[c])
                allocated_now = sum(cell_take.values())
                extras_left = n_real_max - allocated_now

                # distribui o resto pelas maiores partes decimais (desempate por densidade)
                restos = {c: n_extra_ideal[c] - np.floor(n_extra_ideal[c]) for c in sorted_cells}
                sorted_by_resto = sorted(sorted_cells, key=lambda c: (restos[c], n_pts_in_cells[c]), reverse=True)

                ptr = 0
                L = len(sorted_by_resto)
                while extras_left > 0 and L > 0:
                    c = sorted_by_resto[ptr % L]
                    if cell_take[c] < n_pts_in_cells[c]:
                        cell_take[c] += 1
                        extras_left -= 1
                    ptr += 1

        # Relatório final
        n_selected_per_cell = [cell_take[c] for c in cell_take]
        if n_selected_per_cell:
            print(f"[GridFair_density] Distribuição final de pontos/célula:")
            print(f"   min={min(n_selected_per_cell)}, max={max(n_selected_per_cell)}, "
                f"média={np.mean(n_selected_per_cell):.2f}")

        # 4. Seleciona os pontos dentro de cada célula (respeitando cell_take)
        selected_ids = []
        for cell in cell_take:
            m = cell_take[cell]
            if m <= 0:
                continue
            indices = cell_map[cell]
            if type_search == 2:
                pts = coords[indices]
                center = pts.mean(axis=0)
                dists = np.linalg.norm(pts - center, axis=1)
                order = np.argsort(dists)
                chosen = [indices[i] for i in order[:m]]
            elif type_search == 1:
                chosen = indices[:m]
            else:
                perm = np.random.permutation(indices)
                chosen = perm[:m]
            selected_ids.extend(chosen)

        # Por segurança, corta/explode para exatamente n_real_max
        if len(selected_ids) > n_real_max:
            selected_ids = selected_ids[:n_real_max]
        elif len(selected_ids) < n_real_max:
            print(f"[GridFair_density] Aviso: só foi possível selecionar {len(selected_ids)} "
                f"pontos de {n_real_max} desejados.")

        print(f"[GridFair_density] Pontos finais selecionados: {len(selected_ids)} (esperado={n_real_max})")
        t_end = time.time()
        print(f"[GridFair_density] Tempo total de execução: {t_end - t_start:.3f} s\n")
        return [points_real[idx] for idx in selected_ids]


    def _sample_points_gridfair(
        self,
        points_real,
        n_real_max,
        type_search=2,  # 2=medoid/local, 1=primeiro, 0=random
    ):
        import time
        t_start = time.time()
        print("\n[GridFair] Iniciando filtragem justa e proporcional por grid")
        glyph_size = self.glyph_size

        coords = np.array([[p['x'], p['y']] for p in points_real])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        width, height = x_max - x_min, y_max - y_min
        n_cols = math.ceil(width / glyph_size)
        n_rows = math.ceil(height / glyph_size)
        print(f"[GridFair] Grid: {n_rows} rows x {n_cols} cols")

        # 1. Agrupa pontos por célula
        cell_map = {}
        for idx, (x, y) in enumerate(coords):
            col = min(int((x - x_min) / glyph_size), n_cols - 1)
            row = min(int((y - y_min) / glyph_size), n_rows - 1)
            cell_map.setdefault((row, col), []).append(idx)
        print(f"[GridFair] Células não-vazias: {len(cell_map)}")

        # 2. Distribuição inicial justa
        n_cells = len(cell_map)
        base_m = n_real_max // n_cells
        n_pts_in_cells = {key: len(indices) for key, indices in cell_map.items()}
        sorted_cells = sorted(cell_map.keys(), key=lambda c: n_pts_in_cells[c], reverse=True)

        # 3. Inicializa estrutura de quantos pegar em cada célula
        cell_take = {}
        for cell in sorted_cells:
            cell_take[cell] = min(base_m, n_pts_in_cells[cell])

        total_selected = sum(cell_take.values())
        print(f"[GridFair] Alocado inicial: {total_selected} (base_m={base_m}), faltam {n_real_max - total_selected}")

        # 4. Redistribuição dos pontos restantes, sempre para as células mais densas com sobra
        extras_left = n_real_max - total_selected
        while extras_left > 0:
            # Lista só células com pontos sobrando
            cells_with_extra = [c for c in sorted_cells if cell_take[c] < n_pts_in_cells[c]]
            if not cells_with_extra:
                print("[GridFair] Não há mais células com pontos extras disponíveis.")
                break
            # Distribui 1 extra para cada célula mais densa possível
            for cell in cells_with_extra:
                if extras_left == 0:
                    break
                cell_take[cell] += 1
                extras_left -= 1

        print(f"[GridFair] Distribuição final de pontos/célula:")
        n_selected_per_cell = [cell_take[c] for c in sorted_cells]
        print(f"   min={min(n_selected_per_cell)}, max={max(n_selected_per_cell)}, média={np.mean(n_selected_per_cell):.2f}")

        # 5. Para cada célula, escolha exatamente os pontos
        selected_ids = []
        for cell in sorted_cells:
            indices = cell_map[cell]
            m = cell_take[cell]
            if m == 0:
                continue
            if type_search == 2:
                pts = coords[indices]
                center = pts.mean(axis=0)
                dists = np.linalg.norm(pts - center, axis=1)
                order = np.argsort(dists)
                chosen = [indices[i] for i in order[:m]]
            elif type_search == 1:
                chosen = indices[:m]
            else:
                perm = np.random.permutation(indices)
                chosen = perm[:m]
            selected_ids.extend(chosen)


        print(f"[GridFair] Pontos finais selecionados: {len(selected_ids)} (esperado={n_real_max})")
        t_end = time.time()
        print(f"[GridFair] Tempo total de execução: {t_end - t_start:.3f} s\n")
        if len(selected_ids) != n_real_max:
            print(f"[GridFair] Aviso: só foi possível selecionar {len(selected_ids)} pontos de {n_real_max} desejados (dataset não tem mais pontos disponíveis).")
        return [points_real[idx] for idx in selected_ids]
    

    def _assignment_global(self, points, nr_rows, nr_columns):
        return self._grid_rec(points, nr_rows, nr_columns, 0, 0)

    def _grid_rec(self, grid, r, s, i, j):
        size = len(grid)
        if size > 0:
            if size == 1:
                grid[0]['row'] = i
                grid[0]['col'] = j
            else:
                if r > s:
                    half_rows = int(math.ceil(r / 2.0))
                    grid0, grid1 = self._split_grid(grid, min(size, half_rows * s), 'y')
                    self._grid_rec(grid0, half_rows, s, i, j)
                    self._grid_rec(grid1, (r - half_rows), s, (i + half_rows), j)
                else:
                    half_columns = int(math.ceil(s / 2.0))
                    grid0, grid1 = self._split_grid(grid, min(size, half_columns * r), 'x')
                    self._grid_rec(grid0, r, half_columns, i, j)
                    self._grid_rec(grid1, r, (s - half_columns), i, (j + half_columns))
        return grid

    def _split_grid(self, grid, cut_point, direction):
        if direction == 'x':
            grid = sorted(grid, key=lambda cel: (cel['x'], cel['y']))
        else:
            grid = sorted(grid, key=lambda cel: (cel['y'], cel['x']))
        grid0 = grid[:cut_point]
        grid1 = grid[cut_point:]
        return grid0, grid1

    def _gaussian_mask(self, size, sigma):
        mask = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            y = i - size // 2
            for j in range(size):
                x = j - size // 2
                mask[i][j] = 1.0 / (2 * math.pi * sigma * sigma) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        return mask
