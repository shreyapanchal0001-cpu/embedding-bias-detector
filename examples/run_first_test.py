import yaml
import sys
sys.path.insert(0, "..")

from core.detector import BiasDetector
from core.metrics import association_score, disparity_percentage

def main():
    with open("../config/groups.yaml") as f:
        config = yaml.safe_load(f)

    detector = BiasDetector()

    gender = config["gender"]
    a_vecs = detector.embed(gender["group_a"]["terms"])
    b_vecs = detector.embed(gender["group_b"]["terms"])
    centroid_a = detector.group_centroid(a_vecs)
    centroid_b = detector.group_centroid(b_vecs)

    occupations = config["attributes"]["terms"]
    occ_vecs = detector.embed(occupations)

    scores = {}
    for occ, vec in zip(occupations, occ_vecs):
        scores[occ] = association_score(vec, centroid_a, centroid_b)

    overall = disparity_percentage(list(scores.values()))

    print("=" * 55)
    print("EMBEDDING BIAS REPORT — Gender x Occupations")
    print("=" * 55)
    print(f"{'Occupation':<15} {'Score':>8}   Leans Toward")
    print("-" * 55)
    for occ, s in sorted(scores.items(), key=lambda x: x[1]):
        lean = gender["group_a"]["label"] if s > 0 else gender["group_b"]["label"]
        print(f"{occ:<15} {s:>+8.4f}   {lean}")
    print("-" * 55)
    print(f"Overall Disparity: {overall:.1f}%")
    if overall < 20:
        print("Status: [OK]       🟢 Low bias")
    elif overall < 40:
        print("Status: [WARNING]  🟡 Moderate bias")
    else:
        print("Status: [ALERT]    🔴 High bias")

if __name__ == "__main__":
    main()
    