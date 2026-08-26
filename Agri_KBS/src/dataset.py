import numpy as np
import pandas as pd


def sigmoid(x):

    # trasforma il rischio latente in probabilità [0,1]
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(
    n_samples=600,
    random_state=42
):

    # Genero un dataset sintetico per l'agricoltura di precisione.
    # Il target non usa direttamente le regole della Knoewledge Base: 
    # Questo permette di valutare se la conoscenza simbolica fornisca informazione aggiuntiva rispetto ai dati grezzi.

    rng = np.random.default_rng(random_state)

    zone_ids = np.arange(1, n_samples + 1)

    # Variabili ambientali

    temperature = rng.normal(
        loc=23.0,
        scale=5.0,
        size=n_samples
    )

    humidity = np.clip(
        rng.normal(
            loc=72.0,
            scale=15.0,
            size=n_samples
        ),
        30,
        100
    )

    rainfall = np.clip(
        rng.gamma(
            shape=2.0,
            scale=3.0,
            size=n_samples
        ),
        0,
        30
    )

    leaf_wetness = np.clip(
        0.45 * humidity
        + 0.8 * rainfall
        + rng.normal(0, 8, n_samples),
        0,
        100
    )

    wind_speed = np.clip(
        rng.normal(
            loc=8.0,
            scale=4.0,
            size=n_samples
        ),
        0,
        30
    )

    night_temperature = (
        temperature
        - rng.normal(
            4.0,
            2.0,
            n_samples
        )
    )

    crop = rng.choice(
        [
            "Tomato",
            "Grape",
            "Olive"
        ],
        size=n_samples,
        p=[
            0.45,
            0.35,
            0.20
        ]
    )

    # Effetto della coltura

    crop_effect = np.where(
        crop == "Tomato",
        0.45,
        np.where(
            crop == "Grape",
            0.25,
            -0.20
        )
    )

    # Variabile latente

    latent_risk = (
        -4.0
        + 0.045 * humidity
        + 0.035 * leaf_wetness
        + 0.055 * rainfall
        + 0.045 * (temperature - 21.0)
        + crop_effect
        - 0.055 * wind_speed
        + rng.normal(
            0,
            0.8,
            n_samples
        )
    )

    disease_probability = sigmoid(
        latent_risk
    )

    disease = rng.binomial(
        1,
        disease_probability
    )

    df = pd.DataFrame({
        "zone_id": zone_ids,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "leaf_wetness": leaf_wetness,
        "wind_speed": wind_speed,
        "night_temperature": night_temperature,
        "crop": crop,
        "disease": disease
    })

    return df


def save_dataset(df, path):
    """
    Salva il dataset in formato CSV.
    """

    df.to_csv(
        path,
        index=False
    )


if __name__ == "__main__":

    data = generate_dataset()

    print(data.head())

    print(
        "\nNumero record:",
        len(data)
    )

    print(
        "\nDistribuzione del target:"
    )

    print(
        data["disease"].value_counts()
    )