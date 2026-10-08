"""Split-isolation checks and diagnostics."""


def assert_disjoint(name_a, items_a, name_b, items_b):
    overlap = set(items_a) & set(items_b)
    if overlap:
        examples = list(overlap)[:5]
        raise AssertionError(
            f"[DATA LEAK] {name_a} and {name_b} overlap on "
            f"{len(overlap)} items, e.g. {examples}"
        )


def leak_report(
    train_indices_per_client=None,
    val_indices=None,
    test_indices=None,
    label="(indices)",
):
    print(f"\n=== Data Leak Report {label} ===")
    if train_indices_per_client:
        union = set().union(*(set(split) for split in train_indices_per_client))
        total = sum(len(split) for split in train_indices_per_client)
        print(
            f"Clients: {len(train_indices_per_client)} shards | "
            f"sum sizes={total} | union size={len(union)}"
        )
        for left in range(len(train_indices_per_client)):
            for right in range(left + 1, len(train_indices_per_client)):
                overlap = set(train_indices_per_client[left]) & set(
                    train_indices_per_client[right]
                )
                if overlap:
                    print(f"  ⚠️ Client {left + 1} ∩ Client {right + 1} = {len(overlap)}")
    if val_indices is not None:
        print(f"Server-VAL size={len(val_indices)}")
    if test_indices is not None:
        print(f"TEST size={len(test_indices)}")
    print("-------------------------------------------------------------------")
