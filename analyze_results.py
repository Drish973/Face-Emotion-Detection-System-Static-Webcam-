import json

# Load results
with open('detection_results.json', 'r') as f:
    results = json.load(f)

# Calculate accuracy
correct = 0
total = len(results)

for result in results:
    if result['true_emotion'] == result['predicted_emotion']:
        correct += 1

accuracy = correct / total * 100

print(f"Total detections: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

# Per emotion accuracy
from collections import defaultdict
emotion_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

for result in results:
    emotion = result['true_emotion']
    emotion_stats[emotion]['total'] += 1
    if result['true_emotion'] == result['predicted_emotion']:
        emotion_stats[emotion]['correct'] += 1

print("\nPer-emotion accuracy:")
for emotion, stats in emotion_stats.items():
    acc = stats['correct'] / stats['total'] * 100
    print(f"{emotion}: {acc:.2f}% ({stats['correct']}/{stats['total']})")