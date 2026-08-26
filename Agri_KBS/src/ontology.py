from pathlib import Path

import pandas as pd

from owlready2 import (
    get_ontology,
    Thing,
    DataProperty,
    ObjectProperty,
    Imp,
    sync_reasoner
)


ONTOLOGY_DIR = Path("ontology")
ONTOLOGY_DIR.mkdir(
    exist_ok=True
)

ONTOLOGY_PATH = (
    ONTOLOGY_DIR
    / "agriculture.owl"
)


def create_ontology():

    onto = get_ontology(
        "http://example.org/agriculture-kbs.owl"
    )

    with onto:

        # Classi principali
        
        class AgriculturalEntity(Thing):
            pass

        class Crop(AgriculturalEntity):
            pass

        class FieldZone(AgriculturalEntity):
            pass

        class EnvironmentalObservation(
            AgriculturalEntity
        ):
            pass

        class Disease(
            AgriculturalEntity
        ):
            pass

        # RISCHIO

        class RiskCondition(
            AgriculturalEntity
        ):
            pass

        class HighRiskCondition(
            RiskCondition
        ):
            pass

        class MediumRiskCondition(
            RiskCondition
        ):
            pass

        class LowRiskCondition(
            RiskCondition
        ):
            pass

        # Condizioni ambientali

        class HumidityLevel(
            AgriculturalEntity
        ):
            pass

        class HighHumidity(
            HumidityLevel
        ):
            pass

        class NormalHumidity(
            HumidityLevel
        ):
            pass

        class LeafWetnessLevel(
            AgriculturalEntity
        ):
            pass

        class HighLeafWetness(
            LeafWetnessLevel
        ):
            pass

        class NormalLeafWetness(
            LeafWetnessLevel
        ):
            pass

        class TemperatureCondition(
            AgriculturalEntity
        ):
            pass

        class FavorableTemperature(
            TemperatureCondition
        ):
            pass

        class UnfavorableTemperature(
            TemperatureCondition
        ):
            pass

        class WindCondition(
            AgriculturalEntity
        ):
            pass

        class LowWind(
            WindCondition
        ):
            pass

        class HighWind(
            WindCondition
        ):
            pass

        # Malattie

        class DownyMildew(
            Disease
        ):
            pass

        # DATA PROPERTIES

        class hasTemperature(
            DataProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [float]

        class hasHumidity(
            DataProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [float]

        class hasRainfall(
            DataProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [float]

        class hasLeafWetness(
            DataProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [float]

        class hasWindSpeed(
            DataProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [float]

        class hasNightTemperature(
            DataProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [float]

        # Object properties

        class hasCrop(
            ObjectProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [Crop]

        class hasHumidityLevel(
            ObjectProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [HumidityLevel]

        class hasLeafWetnessLevel(
            ObjectProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [LeafWetnessLevel]

        class hasTemperatureCondition(
            ObjectProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [TemperatureCondition]

        class hasWindCondition(
            ObjectProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [WindCondition]

        class hasRiskCondition(
            ObjectProperty
        ):
            domain = [
                EnvironmentalObservation
            ]
            range = [RiskCondition]


        # Individui di dominio

        tomato = Crop(
            "Tomato"
        )

        grape = Crop(
            "Grape"
        )

        olive = Crop(
            "Olive"
        )

        # Regole SWRL

        high_risk_rule = Imp()

        high_risk_rule.set_as_rule(
            """
            EnvironmentalObservation(?x),
            hasHumidityLevel(?x, ?h),
            HighHumidity(?h),
            hasLeafWetnessLevel(?x, ?w),
            HighLeafWetness(?w),
            hasTemperatureCondition(?x, ?t),
            FavorableTemperature(?t)
            ->
            HighRiskCondition(?x)
            """
        )

        medium_risk_rule = Imp()

        medium_risk_rule.set_as_rule(
            """
            EnvironmentalObservation(?x),
            hasHumidityLevel(?x, ?h),
            HighHumidity(?h),
            hasTemperatureCondition(?x, ?t),
            FavorableTemperature(?t)
            ->
            MediumRiskCondition(?x)
            """
        )

    return onto


def classify_environment(row):

    humidity = float(
        row["humidity"]
    )

    leaf_wetness = float(
        row["leaf_wetness"]
    )

    temperature = float(
        row["temperature"]
    )

    wind = float(
        row["wind_speed"]
    )

    # Umidità

    humidity_level = (
        "HighHumidity"
        if humidity >= 80
        else "NormalHumidity"
    )

    # Bagnatura fogliare

    wetness_level = (
        "HighLeafWetness"
        if leaf_wetness >= 70
        else "NormalLeafWetness"
    )

    # Temperatura

    temperature_condition = (
        "FavorableTemperature"
        if 18 <= temperature <= 30
        else "UnfavorableTemperature"
    )

    # Vento

    wind_condition = (
        "LowWind"
        if wind <= 15
        else "HighWind"
    )

    return (
        humidity_level,
        wetness_level,
        temperature_condition,
        wind_condition
    )


def apply_ontology_reasoning(df):

    print(
        "\n=========================================="
    )

    print(
        "FASE 1 - KNOWLEDGE BASE + REASONING"
    )

    print(
        "=========================================="
    )

    onto = create_ontology()

    with onto:

        for _, row in df.iterrows():

            observation = (
                onto.EnvironmentalObservation(
                    f"Observation_{int(row['zone_id'])}"
                )
            )

            # Valori numerici

            observation.hasTemperature = [
                float(row["temperature"])
            ]

            observation.hasHumidity = [
                float(row["humidity"])
            ]

            observation.hasRainfall = [
                float(row["rainfall"])
            ]

            observation.hasLeafWetness = [
                float(row["leaf_wetness"])
            ]

            observation.hasWindSpeed = [
                float(row["wind_speed"])
            ]

            observation.hasNightTemperature = [
                float(row["night_temperature"])
            ]

            # Coltura

            crop_name = row["crop"]

            if crop_name == "Tomato":

                observation.hasCrop = [
                    onto.Tomato
                ]

            elif crop_name == "Grape":

                observation.hasCrop = [
                    onto.Grape
                ]

            else:

                observation.hasCrop = [
                    onto.Olive
                ]

            # Categorie semantiche

            (
                humidity_level,
                wetness_level,
                temperature_condition,
                wind_condition
            ) = classify_environment(row)

            # Umidità

            humidity_class = onto[
                humidity_level
            ]

            humidity_instance = (
                humidity_class(
                    f"{humidity_level}_{int(row['zone_id'])}"
                )
            )

            observation.hasHumidityLevel = [
                humidity_instance
            ]

            # Bagnatura fogliare

            wetness_class = onto[
                wetness_level
            ]

            wetness_instance = (
                wetness_class(
                    f"{wetness_level}_{int(row['zone_id'])}"
                )
            )

            observation.hasLeafWetnessLevel = [
                wetness_instance
            ]

            # Temperatura

            temperature_class = onto[
                temperature_condition
            ]

            temperature_instance = (
                temperature_class(
                    f"{temperature_condition}_{int(row['zone_id'])}"
                )
            )

            observation.hasTemperatureCondition = [
                temperature_instance
            ]

            # Vento

            wind_class = onto[
                wind_condition
            ]

            wind_instance = (
                wind_class(
                    f"{wind_condition}_{int(row['zone_id'])}"
                )
            )

            observation.hasWindCondition = [
                wind_instance
            ]

    # Reasoning 

    print(
        "Avvio HermiT Reasoner..."
    )

    with onto:

        sync_reasoner(
            infer_property_values=True
        )

    print(
        "Reasoning completato."
    )

    # ESTRAZIONE CONOSCENZA DEDOTTA

    high_risk = []
    medium_risk = []

    for _, row in df.iterrows():

        observation = onto[
            f"Observation_{int(row['zone_id'])}"
        ]

        high = (
            onto.HighRiskCondition
            in observation.INDIRECT_is_a
        )

        medium = (
            onto.MediumRiskCondition
            in observation.INDIRECT_is_a
        )

        high_risk.append(
            int(high)
        )

        medium_risk.append(
            int(medium)
        )

    enriched = df.copy()

    enriched[
        "bk_high_risk"
    ] = high_risk

    enriched[
        "bk_medium_risk"
    ] = medium_risk

    # Salvataggio ontologia

    onto.save(
        file=str(ONTOLOGY_PATH),
        format="rdfxml"
    )

    print(
        f"Ontologia salvata in: "
        f"{ONTOLOGY_PATH}"
    )

    print(
        "High-risk dedotte:",
        sum(high_risk)
    )

    print(
        "Medium-risk dedotte:",
        sum(medium_risk)
    )

    return enriched