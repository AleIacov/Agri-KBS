import random
import time


# Dominio dei Valori 

TIME_SLOTS = [
    8,
    10,
    12,
    14,
    16,
    18
]


# Funzione obiettivo 

def objective(solution, zone_data):

    # funzione di costo (soft constraints): valuto ritardo e penalizzo slot serali

    total_cost = 0.0

    for zone, slot in solution.items():

        data = zone_data[zone]

        ideal_time = data["ideal_time"]

        # Scostamento dall'orario ideale
        delay = abs(slot - ideal_time)

        # Costo principale: 2 unità per ogni ora di scostamento

        total_cost += 2.0 * delay

        # Penalizzazione per trattamenti effettuati nel pomeriggio.
        if slot >= 16:
            total_cost += 1.0

    return total_cost


# Vincoli

def violates_constraints(
    zone,
    slot,
    assignment,
    zone_data
):

    # controllo tutti gli hard constraints del CSP

    # Vincolo 1: un solo drone

    for other_zone, other_slot in assignment.items():

        if (
            other_zone != zone
            and other_slot == slot
        ):
            return True

    # Vincolo 2: zone adiacenti 

    adjacent = zone_data[zone]["adjacent"]

    for other_zone in adjacent:

        if other_zone in assignment:

            other_slot = assignment[other_zone]

            # Le due zone devono essere distanti
            # almeno due ore.
            if abs(slot - other_slot) < 2:
                return True

    # Vincolo 3: vento 

    if zone_data[zone]["wind_speed"] > 20:
        return True

    return False


# BACKTRACKING + FORWARD CHECKING

def backtracking_search(
    zones,
    zone_data
):
    # ricerca sistematica con Forward Checking e euristica MRV

    # Costruzione dei domini 
    domains = {
        zone: [
            slot
            for slot in TIME_SLOTS
            if zone_data[zone]["wind_speed"] <= 20
        ]
        for zone in zones
    }

    nodes = 0

    best_solution = None
    best_cost = float("inf")

    def forward_check(
        assignment,
        remaining_domains
    ):
        nonlocal nodes
        nonlocal best_solution
        nonlocal best_cost

        nodes += 1

        # Soluzione completa 

        if len(assignment) == len(zones):

            current_cost = objective(
                assignment,
                zone_data
            )

            if current_cost < best_cost:

                best_cost = current_cost
                best_solution = assignment.copy()

            return

        # MRV

        unassigned = [
            zone
            for zone in zones
            if zone not in assignment
        ]

        selected = min(
            unassigned,
            key=lambda zone: len(
                remaining_domains[zone]
            )
        )

        # Esplorazione del dominio

        for value in remaining_domains[selected]:

            if violates_constraints(
                selected,
                value,
                assignment,
                zone_data
            ):
                continue

            assignment[selected] = value

            # Copia dei domini per il nuovo livello della ricerca
            new_domains = {
                zone: values.copy()
                for zone, values
                in remaining_domains.items()
            }

            consistent = True

            # FORWARD CHECKING

            for other in unassigned:

                if other == selected:
                    continue

                filtered = []

                for candidate in new_domains[other]:

                    if not violates_constraints(
                        other,
                        candidate,
                        assignment,
                        zone_data
                    ):
                        filtered.append(candidate)

                new_domains[other] = filtered

                # Se un dominio diventa vuoto, il ramo non può produrre una soluzione
                if not filtered:

                    consistent = False
                    break

            if consistent:

                forward_check(
                    assignment,
                    new_domains
                )

            # Backtracking
            del assignment[selected]

    forward_check(
        {},
        domains
    )

    return (
        best_solution,
        nodes
    )


# Conteggio dei conflitti 

def count_conflicts(
    solution,
    zone_data
):

    # metrica per la ricerca locale

    conflicts = 0

    zones = list(solution.keys())

    # CONFLITTI TRA COPPIE DI ZONE

    for i in range(len(zones)):

        z1 = zones[i]

        for j in range(i + 1, len(zones)):

            z2 = zones[j]

            # Stesso intervallo temporale.
            if solution[z1] == solution[z2]:

                conflicts += 1

            # Zone adiacenti
            if z2 in zone_data[z1]["adjacent"]:

                if abs(
                    solution[z1] - solution[z2]
                ) < 2:

                    conflicts += 1

    # Vincolo del vento

    for zone in zones:

        if zone_data[zone]["wind_speed"] > 20:

            conflicts += 1

    return conflicts


# MIN-CONFLICTS

def min_conflicts(
    zones,
    zone_data,
    max_steps=1000,
    random_state=42
):

    # euristica stocastica locale, scambia slot per riparare violazioni

    rng = random.Random(random_state)

    solution = {}

    # Soluzione iniziale casuale 

    for zone in zones:

        valid_slots = [
            slot
            for slot in TIME_SLOTS
            if zone_data[zone]["wind_speed"] <= 20
        ]

        if not valid_slots:

            return (
                None,
                max_steps
            )

        solution[zone] = rng.choice(valid_slots)

    # Stato inizile 

    best_solution = solution.copy()

    best_conflicts = count_conflicts(
        solution,
        zone_data
    )

    best_cost = objective(
        solution,
        zone_data
    )

    # Ricerca locale

    for step in range(max_steps):

        conflicts = count_conflicts(
            solution,
            zone_data
        )

        # Soluzione ammissibile trovata.
        if conflicts == 0:

            return (
                solution,
                step + 1
            )

        conflicted = []

        # Individuazione delle variabili in conflitto

        for zone in zones:

            other_assignment = {
                z: s
                for z, s in solution.items()
                if z != zone
            }

            if any(
                violates_constraints(
                    zone,
                    slot,
                    other_assignment,
                    zone_data
                )
                for slot in TIME_SLOTS
                if zone_data[zone]["wind_speed"] <= 20
            ):

                conflicted.append(zone)

        if not conflicted:

            return (
                best_solution,
                step + 1
            )

        # Scelta casuale della zona in conflitto.
        zone = rng.choice(conflicted)

        # Valori candidati

        valid_slots = [

            slot

            for slot in TIME_SLOTS

            if (
                zone_data[zone]["wind_speed"] <= 20
                and
                not violates_constraints(
                    zone,
                    slot,
                    {
                        z: s
                        for z, s in solution.items()
                        if z != zone
                    },
                    zone_data
                )
            )
        ]

        if not valid_slots:
            continue

        candidate_costs = []

        for slot in valid_slots:

            candidate = solution.copy()
            candidate[zone] = slot

            candidate_conflicts = count_conflicts(
                candidate,
                zone_data
            )

            candidate_cost = objective(
                candidate,
                zone_data
            )

            candidate_costs.append(
                (
                    candidate_conflicts,
                    candidate_cost,
                    slot
                )
            )

        # Prima si minimizzano i conflitti; a parità di conflitti si minimizza il costo
        (
            candidate_conflicts,
            candidate_cost,
            best_slot
        ) = min(candidate_costs)

        solution[zone] = best_slot

        current_conflicts = count_conflicts(
            solution,
            zone_data
        )

        current_cost = objective(
            solution,
            zone_data
        )

        if (
            current_conflicts < best_conflicts
            or (
                current_conflicts == best_conflicts
                and current_cost < best_cost
            )
        ):

            best_solution = solution.copy()
            best_conflicts = current_conflicts
            best_cost = current_cost

    return (
        best_solution,
        max_steps
    )


# Costruzione del problema CSP

def build_zone_problem(
    df,
    predicted_zones
):
 
    # preparo il CSP prendendo le zone marchiate dal ML
    selected = df[
        df["zone_id"].isin(predicted_zones)
    ].copy()

    # Il problema sperimentale considera al massimo quattro zone a rischio.
    selected = selected.head(4)

    zones = [
        f"Zone_{int(zone_id)}"
        for zone_id in selected["zone_id"]
    ]

    zone_data = {}

    for _, row in selected.iterrows():

        zone = f"Zone_{int(row['zone_id'])}"

        zone_data[zone] = {
            "wind_speed": float(
                row["wind_speed"]
            ),
            "ideal_time": 10,
            "adjacent": set()
        }

    # assumo adiacenza lineare stile pipeline

    for i, zone in enumerate(zones):

        if i > 0:

            zone_data[zone]["adjacent"].add(
                zones[i - 1]
            )

        if i < len(zones) - 1:

            zone_data[zone]["adjacent"].add(
                zones[i + 1]
            )

    return (
        zones,
        zone_data
    )


# Esecuzione del CSP

def run_csp(
    df,
    predicted_zones
):

    print(
        "\n=========================================="
    )

    print(
        "FASE 3 - CONSTRAINT SATISFACTION PROBLEM"
    )

    print(
        "=========================================="
    )

    zones, zone_data = build_zone_problem(
        df,
        predicted_zones
    )

    if len(zones) == 0:

        print(
            "Nessuna zona disponibile per il CSP."
        )

        return {
            "backtracking_solution": None,
            "backtracking_nodes": 0,
            "backtracking_time": 0.0,
            "backtracking_cost": None,
            "min_conflicts_solution": None,
            "min_conflicts_steps": 0,
            "min_conflicts_time": 0.0,
            "min_conflicts_cost": None
        }

    print(
        "Zone da trattare:",
        zones
    )

    # BACKTRACKING + FORWARD CHECKING

    start = time.perf_counter()

    (
        bt_solution,
        bt_nodes
    ) = backtracking_search(
        zones,
        zone_data
    )

    bt_time = (
        time.perf_counter()
        - start
    )

    bt_cost = None

    if bt_solution is not None:

        bt_cost = objective(
            bt_solution,
            zone_data
        )

    print(
        "\nBacktracking + Forward Checking"
    )

    print(
        "Soluzione:",
        bt_solution
    )

    print(
        "Costo:",
        bt_cost
    )

    print(
        "Nodi esplorati:",
        bt_nodes
    )

    print(
        "Tempo:",
        f"{bt_time:.6f}",
        "secondi"
    )

    # MIN-CONFLICTS

    start = time.perf_counter()

    (
        mc_solution,
        mc_steps
    ) = min_conflicts(
        zones,
        zone_data,
        max_steps=1000,
        random_state=42
    )

    mc_time = (
        time.perf_counter()
        - start
    )

    mc_cost = None

    if mc_solution is not None:

        mc_cost = objective(
            mc_solution,
            zone_data
        )

    print(
        "\nMin-Conflicts"
    )

    print(
        "Soluzione:",
        mc_solution
    )

    print(
        "Costo:",
        mc_cost
    )

    print(
        "Iterazioni:",
        mc_steps
    )

    print(
        "Tempo:",
        f"{mc_time:.6f}",
        "secondi"
    )

    # RISULTATI

    return {
        "backtracking_solution":
            bt_solution,

        "backtracking_nodes":
            bt_nodes,

        "backtracking_time":
            bt_time,

        "backtracking_cost":
            bt_cost,

        "min_conflicts_solution":
            mc_solution,

        "min_conflicts_steps":
            mc_steps,

        "min_conflicts_time":
            mc_time,

        "min_conflicts_cost":
            mc_cost
    }