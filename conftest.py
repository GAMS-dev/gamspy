from pytest import Function


def pytest_collection_modifyitems(items: list[Function]) -> None:
    """
    Enforces the order of tests. GAMSPy currently has the following marks:

    - "unit: unit tests"
    - "integration: integration tests"
    - "cli: cli tests"
    - "doc: doctests"
    - "engine: engine tests"
    - "neos: neos tests"
    - "model_library: run all model library"

    We want to run the unit tests first and model_library tests last.
    """
    all_markers = []
    for item in items:
        markers = [marker.name for marker in item.iter_markers()]
        all_markers.append(markers)

    # Run unit tests first
    new_items: list[Function] = []
    model_library = None
    for item, markers in zip(items, all_markers):
        if "model_library" in markers:
            model_library = item
            continue

        if "unit" in markers:
            new_items.insert(0, item)
        else:
            new_items.append(item)

    # Run model library last since takes a lot of time
    if model_library is not None:
        new_items.append(model_library)

    items[:] = new_items
