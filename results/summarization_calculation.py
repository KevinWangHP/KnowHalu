import json
import matplotlib.pyplot as plt
import numpy as np


def metric_calculation(judgment_file):
    # Load the JSON file
    with open(judgment_file, 'r') as file:
        data = json.load(file)

    # Initialize counters
    total = 0
    correct_count = 0
    incorrect_count = 0
    inconclusive_count = 0
    no_answer = 0

    # Iterate over the dataset and count judgments
    for item in data:
        total += 1
        flag = True
        for part in item:
            judgement = part[0]
            if "INCONCLUSIVE" in judgement:
                flag = False
            elif "INCORRECT" in judgement:
                flag = False
        if flag:
            correct_count += 1
        else:
            incorrect_count += 1


    # Print the results
    print(f"TOTAL: {total}")
    print(f"CORRECT: {correct_count}")
    print(f"INCORRECT: {incorrect_count}")
    print(f"INCONCLUSIVE: {inconclusive_count}")
    print(f"NO_ANSWER: {no_answer}")
    return total, correct_count, incorrect_count, inconclusive_count, no_answer


def calculate_metrics(hallucinated_res, right_res):
    total_h, correct_h, incorrect_h, inconclusive_h, no_answer_h = hallucinated_res
    total_r, correct_r, incorrect_r, inconclusive_r, no_answer_r = right_res

    tn = correct_r  # True Positives (CORRECT in right_res)
    tp = incorrect_h  # True Negatives (INCORRECT in hallucinated_res)

    # Avoid division by zero
    tnr = tn / total_r if total_r > 0 else 0  # Sensitivity or Recall
    tpr = tp / total_h if total_h > 0 else 0  # Specificity
    accuracy = (tp + tn) / (total_h + total_r) if (total_h + total_r) > 0 else 0

    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return tpr, tnr, accuracy


# hallucinated_res = metric_calculation('./qa/judgment/Starling-LM-7B-alpha/hallucinated_ground_semantic.json')
right_res = metric_calculation('./summarization/judgment/Starling-LM-7B-alpha/right_semantic_top2.json')
hallucinated_res = metric_calculation('./summarization/judgment/Starling-LM-7B-alpha/hallucinated_semantic_top2.json')


tpr, tnr, accuracy = calculate_metrics(hallucinated_res, right_res)


hallucinated_values = [hallucinated_res[1], hallucinated_res[3], hallucinated_res[2]]
right_values = [right_res[1], right_res[3], right_res[2]]
total_h = hallucinated_res[0]
total_r = right_res[0]

categories = ["Correct", "Inconclusive", "Incorrect"]
x = np.arange(len(categories))  # Label locations

hallucinated_percentages = [(v / total_h) * 100 if total_h > 0 else 0 for v in hallucinated_values]
right_percentages = [(v / total_r) * 100 if total_r > 0 else 0 for v in right_values]

# Plot Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35

rects1 = ax.bar(x - bar_width/2, hallucinated_percentages, bar_width, label="Hallucinated")
rects2 = ax.bar(x + bar_width/2, right_percentages, bar_width, label="Right")

# Annotate bars with their percentages
for rect, perc in zip(rects1, hallucinated_percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, f'{perc:.1f}%', ha='center', va='bottom')

for rect, perc in zip(rects2, right_percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, f'{perc:.1f}%', ha='center', va='bottom')

ax.set_xlabel("Judgment Categories")
ax.set_ylabel("Percentage (%)")
ax.set_title("Structured (Hallucinated vs Right)")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.show()



