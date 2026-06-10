from qcom.data import CountsData, combine_datasets, normalize_to_probabilities, sample_data


counts = CountsData({"00": 50, "01": 25, "10": 25})
probabilities = normalize_to_probabilities(counts)
sampled = sample_data(counts.to_dict(), total_count=counts.shots, sample_size=20)
merged = combine_datasets(CountsData(sampled), CountsData(sampled), return_data=True)

print(probabilities)
print(merged.shots)
