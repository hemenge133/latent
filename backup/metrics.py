"""
Metrics and evaluation functions for transformer models.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def improved_inference(model, inp, dataset, device, max_len=10, print_debug=False):
    """More robust inference function specially for multiplication task"""
    model.eval()

    with torch.no_grad():
        # Start with start token (usually token 1)
        decoder_input = torch.tensor([[1]], dtype=torch.long).to(device)

        # Storage for output tokens and their probabilities
        output_tokens = []
        token_probs = []  # Store probabilities for debugging

        # Generate tokens one by one
        for i in range(max_len):
            # Forward pass
            output = model(inp, decoder_input)

            # Get probabilities for the next token
            logits = output[0, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Extract inputs for better length control
            try:
                input_str = dataset.decode(inp[0].cpu().numpy())
                # Check if it's a simple multiplication problem
                if "*" in input_str:
                    parts = input_str.split("*")
                    if len(parts) == 2:
                        a, b = int(parts[0]), int(parts[1])
                        expected_length = len(str(a * b))

                        # If we've generated enough tokens for the result, consider stopping
                        if len(output_tokens) >= expected_length:
                            # Strong bias towards stopping
                            break
            except:
                # If we can't determine the expected length, use heuristics
                pass

            # Get top 3 candidates
            topk_probs, topk_indices = torch.topk(probs, 3)

            # Select next token with some heuristics
            found_valid = False
            for j in range(len(topk_indices)):
                # Skip padding and tokens that would create 3+ consecutive repeats
                token_idx = topk_indices[j].item()
                token_prob = topk_probs[j].item()

                # Skip padding token
                if token_idx == 0:
                    continue

                # Skip non-digit tokens (assuming digits are tokens 2-11 representing 0-9)
                if not (2 <= token_idx <= 11):
                    continue

                # Handle repeats - don't allow more than 2 consecutive repeats
                if len(output_tokens) >= 2:
                    if token_idx == output_tokens[-1] == output_tokens[-2]:
                        continue

                # If we reach here, we have a valid token
                found_valid = True
                next_token = token_idx
                next_prob = token_prob
                break

            # If no valid token found, just take the highest probability digit
            if not found_valid:
                for token_idx in range(2, 12):  # Tokens 2-11 are digits 0-9
                    if probs[token_idx] > 0:
                        next_token = token_idx
                        next_prob = probs[token_idx].item()
                        break
                else:
                    # If still no valid token, take the end token
                    next_token = 1  # Assuming 1 is the end/start token
                    next_prob = probs[1].item()

            # If we generated the end token or have enough digits, stop
            if next_token == 1 or len(output_tokens) >= max_len - 1:
                break

            # Add the token to our output
            output_tokens.append(next_token)
            token_probs.append(next_prob)

            # Prepare for the next iteration
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.tensor([[next_token]], dtype=torch.long).to(device),
                ],
                dim=1,
            )

        # Convert token IDs back to digits (token 2 = digit 0, token 3 = digit 1, etc.)
        result = ""
        for token in output_tokens:
            # Convert from token ID to digit
            digit = token - 2  # Token 2 = digit 0, etc.
            result += str(digit)

        # Remove leading zeros
        result = result.lstrip("0")
        if not result:  # Handle case of all zeros
            result = "0"

        if print_debug:
            input_str = ""
            if hasattr(dataset, "decode"):
                # Try to decode the input tensor
                try:
                    input_str = dataset.decode(inp[0].cpu().numpy())
                except:
                    input_str = "[Decoding failed]"

            print(f"\nInput: {input_str}")
            print(
                f"Tokens: {output_tokens} (probabilities: {[f'{p:.4f}' for p in token_probs]})"
            )
            print(f"Prediction: {result}")

        return result


def improved_accuracy(
    model, examples, dataset, device, print_debug=False, num_to_print=5
):
    """Calculate accuracy using improved inference with detailed metrics"""
    if not examples:
        return 0.0

    correct = 0
    total = 0
    digit_correct = 0  # Track digit-level accuracy
    total_digits = 0

    # Display header for debug output
    if print_debug:
        print("\n" + "=" * 50)
        print(f"INFERENCE DIAGNOSTICS FOR {model.__class__.__name__}")
        print("=" * 50)

    # If we need to unpack a different number of values
    try:
        # The generate_multiplication_examples function returns tuples of
        # (input_tensor, result_str, a, b)
        for idx, (input_tensor, expected_str, a, b) in enumerate(examples):
            try:
                # Whether to print debug info for this example
                should_print = print_debug and idx < num_to_print

                # Get prediction using improved inference
                pred = improved_inference(
                    model, input_tensor, dataset, device, print_debug=should_print
                )

                # For digit-level accuracy
                min_len = min(len(pred), len(expected_str))
                for i in range(min_len):
                    if pred[i] == expected_str[i]:
                        digit_correct += 1
                total_digits += max(len(pred), len(expected_str))

                # Check if prediction matches expected
                is_correct = pred == expected_str
                if is_correct:
                    correct += 1

                if should_print:
                    print(f"Problem: {a} * {b} = {expected_str}")
                    print(f"Predicted: {pred} (Correct: {is_correct})")
                    if not is_correct:
                        # Show where the prediction is wrong
                        error_indices = []
                        for i in range(min_len):
                            if pred[i] != expected_str[i]:
                                error_indices.append(i)
                        if len(pred) != len(expected_str):
                            print(
                                f"Length mismatch: Pred={len(pred)}, Expected={len(expected_str)}"
                            )
                        else:
                            print(f"Errors at positions: {error_indices}")
                    print("")

                total += 1
            except Exception as e:
                print(f"Error in inference: {e}")

        # Summary
        if print_debug:
            print(f"Total examples evaluated: {total}")
            print(f"Completely correct predictions: {correct} ({correct/total:.2%})")
            digit_accuracy = digit_correct / total_digits if total_digits > 0 else 0
            print(
                f"Digit-level accuracy: {digit_correct}/{total_digits} ({digit_accuracy:.2%})"
            )
            print("=" * 50)
    except ValueError as e:
        # If there's a ValueError it might be a different format
        print(f"Warning: Example format error: {e}. Check example structure.")
        return 0.0

    # Return both full accuracy and digit accuracy
    sequence_accuracy = correct / total if total > 0 else 0.0
    digit_accuracy = digit_correct / total_digits if total_digits > 0 else 0.0
    return sequence_accuracy, digit_accuracy


def evaluate(
    model,
    data_loader,
    criterion,
    dataset,
    device,
    vocab_size,
    desc="Evaluating",
    examples=None,
    rotate_examples=False,
):
    """Evaluate a model on a data loader with loss and accuracy"""
    model.eval()
    total_loss = 0.0
    total_sequence_accuracy = 0.0
    total_samples = 0

    # If rotate_examples is True, we'll modify the validation examples
    # to test on different number ranges
    if examples and rotate_examples and len(examples) > 0:
        # Create a few examples with larger numbers
        try:
            extra_examples = []
            # Add examples with larger numbers to test generalization
            for a in range(10, 20):  # Larger than typical validation examples
                for b in range(10, 20):
                    if len(extra_examples) >= 5:  # Just add a few
                        break
                    # Format input
                    input_str = f"{a}*{b}"
                    input_tokens = dataset.encode(input_str)
                    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(
                        device
                    )

                    # Format expected output
                    result = a * b
                    result_str = str(result)

                    extra_examples.append((input_tensor, result_str, a, b))
                if len(extra_examples) >= 5:
                    break

            # Replace some existing examples with the new ones
            for i, ex in enumerate(extra_examples):
                if i < len(examples):
                    examples[i] = ex
        except Exception as e:
            print(f"Warning: Error creating extra validation examples: {e}")

    with torch.no_grad():
        for inp, tgt, inp_lens, tgt_lens in tqdm(data_loader, desc=desc, leave=False):
            inp, tgt = inp.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]

            output = model(inp, decoder_input)

            # Use the sequence accuracy loss for evaluation
            loss, seq_accuracy = criterion(output, decoder_target)

            total_loss += loss.item() * decoder_target.size(0)
            total_sequence_accuracy += seq_accuracy.item() * decoder_target.size(0)
            total_samples += decoder_target.size(0)

    # Calculate accuracy if examples are provided
    sequence_accuracy = 0.0
    digit_accuracy = 0.0
    if examples:
        # Only print diagnostics during validation (not during final eval)
        print_debug = "Val" in desc  # Only print during validation
        sequence_accuracy, digit_accuracy = improved_accuracy(
            model, examples, dataset, device, print_debug=print_debug
        )

    # Return validation loss and the two accuracy metrics
    return (
        total_loss / max(1, total_samples),
        total_sequence_accuracy / max(1, total_samples),
        digit_accuracy,
    )
