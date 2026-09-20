"""Simulate trajectories, classify them, print accuracy and feature ranking."""

from spt_classify import classify_tracks, make_dataset

if __name__ == "__main__":
    tracks, labels = make_dataset(n_per_class=250, n_steps=100, seed=0)
    print(f"{len(tracks)} trajectories, {len(set(labels))} classes")

    result = classify_tracks(tracks, labels, cv=5)
    print(f"\n5-fold accuracy: {result['accuracy_mean']:.3f} +/- {result['accuracy_std']:.3f}")

    print("\nFeature ranking (mutual information):")
    for name, score in result["feature_ranking"]:
        print(f"  {name:<20s} {score:.3f}")
