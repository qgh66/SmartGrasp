# Run one model
python test.py --root sample_data --model gpt-4o --closed-loop

# Run 4 models in sequence
for M in gpt-4o gpt-5.4-mini gpt-5.5 qwen3-vl-plus; do
    python test.py --root sample_data --model "$M" --closed-loop
done

# Debug with only 2 scenes
python test.py --root sample_data --model gpt-4o --closed-loop --limit 2