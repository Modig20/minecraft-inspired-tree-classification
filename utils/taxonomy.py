CATEGORY_SPECIES = {
    "Oak": ["quercus robur", "quercus alba"],
    "Dark Oak": ["quercus velutina", "quercus rubra"],
    "Pale Oak": ["quercus alba", "quercus lobata"],
    "Spruce": ["picea abies", "picea sitchensis"],
    "Birch": ["betula pendula", "betula papyrifera"],
    "Poplar": ["populus tremuloides", "populus alba"],
    "Jungle": ["ceiba pentandra", "mahogany", "theobroma cacao"],
    "Acacia": ["vachellia tortilis"],
    "Mangrove": ["rhizophora mangle"],
    "Cherry": ["prunus yedoensis", "prunus subhirtella"],
}

DEFAULT_SPECIES = [
    "quercus robur",
    "quercus alba",
    "quercus velutina",
    "quercus rubra",
    "picea abies",
    "populus tremuloides",
    "prunus yedoensis",
    "prunus subhirtella",
]


def display_species(species):
    words = species.replace("_", " ").split()
    if not words:
        return species
    return " ".join([words[0].capitalize(), *[word.lower() for word in words[1:]]])


def categories_for_species(species):
    return [
        category
        for category, species_names in CATEGORY_SPECIES.items()
        if species in species_names
    ]


def available_categories(species_names):
    available_species = set(species_names)
    return [
        category
        for category, category_species in CATEGORY_SPECIES.items()
        if any(species in available_species for species in category_species)
    ]


def dataset_category_coverage(class_names):
    available = set(class_names)
    rows = []
    for category, listed_species in CATEGORY_SPECIES.items():
        present = [name for name in listed_species if name in available]
        missing = [name for name in listed_species if name not in available]
        if not present:
            status = "Unavailable"
        elif missing:
            status = "Partial"
        else:
            status = "Available"
        rows.append({
            "Category": category,
            "Available species": ", ".join(display_species(name) for name in present) or "None",
            "Not in dataset": ", ".join(display_species(name) for name in missing) or "None",
            "Coverage": status,
        })
    return rows
