"""
Hii!
Welcome to NeuralGates demo :)

This script demonstrates how to use the NeuralGates to create and train
neural networks that behave like logical gates and digital circuits.

"""

import matplotlib.pyplot as plt
import time

# Import the Neural Logic Gates library
# In a real scenario, you would have installed this as a package.
# For this example, we assume the library is in the same directory.
from neural_logic_gates import NeuralGate, HalfAdder, FullAdder, LOGIC_GATES

# Set to True to see detailed output during training
VERBOSE = True
# Set random seed for reproducibility (optional)
SEED = 42

#######################
# Part 1: Basic Logic Gates
#######################

def basic_logic_gates_example():
    """
    This function demonstrates how to create and train individual logic gates.
    It covers the basics of neural network training for logical operations.
    """
    print("\n" + "="*50)
    print("PART 1: BASIC LOGIC GATES")
    print("="*50)
    
    # Let's create and train an AND gate
    print("\n1. Creating and training an AND gate:")
    print("-" * 40)
    
    # Initialize a neural gate for the AND operation
    # The gate_type parameter automatically sets up the appropriate training data
    and_gate = NeuralGate(gate_type="AND", seed=SEED)
    
    # Before training, let's see the initial random performance
    print("\nPerformance BEFORE training:")
    and_gate.evaluate()
    
    # Train the model
    # - epochs: number of training iterations
    # - learning_rate: controls how much the weights are adjusted in each step
    # - verbose: whether to print progress during training
    # - log_interval: how often to record the cost for plotting
    print("\nTraining the AND gate...")
    and_gate.train(epochs=5000, learning_rate=0.1, verbose=VERBOSE, log_interval=500)
    
    # After training, let's see the improved performance
    print("\nPerformance AFTER training:")
    and_gate.evaluate()
    
    # We can also visualize the training progress
    print("\nPlotting training history...")
    and_gate.plot_training_history()
    
    # Let's see the learned parameters
    params = and_gate.get_parameters()
    print("\nLearned parameters:")
    for param, value in params.items():
        print(f"{param}: {value:.6f}")
    
    # Now, let's create a more complex gate: XOR
    # XOR is not linearly separable, making it more challenging to learn
    print("\n2. Creating and training an XOR gate:")
    print("-" * 40)
    
    # Initialize a neural gate for the XOR operation
    xor_gate = NeuralGate(gate_type="XOR", seed=SEED)
    
    # Train the model with more epochs due to increased complexity
    print("\nTraining the XOR gate...")
    xor_gate.train(epochs=10000, learning_rate=0.1, verbose=VERBOSE, log_interval=1000)
    
    # Evaluate the XOR gate after training
    print("\nXOR gate performance after training:")
    xor_gate.evaluate()
    xor_gate.plot_training_history()
    
    # Demonstrate how to use the trained gates for prediction
    print("\n3. Making predictions with trained gates:")
    print("-" * 40)
    
    # Test inputs
    test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for x1, x2 in test_cases:
        # The forward method returns the raw output (continuous between -1 and 1)
        raw_and = and_gate.forward(x1, x2)
        # The predict method returns the binary prediction (0 or 1)
        pred_and = and_gate.predict(x1, x2)
        
        raw_xor = xor_gate.forward(x1, x2)
        pred_xor = xor_gate.predict(x1, x2)
        
        print(f"Inputs ({x1}, {x2}):")
        print(f"  AND gate - Raw output: {raw_and:.4f}, Prediction: {pred_and}")
        print(f"  XOR gate - Raw output: {raw_xor:.4f}, Prediction: {pred_xor}")
    
    return and_gate, xor_gate


#######################
# Part 2: Custom Logic Gates
#######################

def custom_logic_gate_example():
    """
    This function demonstrates how to create and train custom logic gates
    with user-defined truth tables.
    """
    print("\n" + "="*50)
    print("PART 2: CUSTOM LOGIC GATES")
    print("="*50)
    
    # Define a custom truth table
    # This example creates an IMPLICATION gate (A → B)
    # Truth table: A | B | A→B
    #              0 | 0 | 1
    #              0 | 1 | 1
    #              1 | 0 | 0
    #              1 | 1 | 1
    implication_data = [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ]
    
    # Create a neural gate without specifying a predefined type
    implication_gate = NeuralGate(seed=SEED)
    
    # Train on the custom truth table
    print("\nTraining a custom IMPLICATION gate...")
    implication_gate.train(epochs=5000, learning_rate=0.1, 
                          training_data=implication_data, 
                          verbose=VERBOSE, log_interval=500)
    
    # Evaluate the custom gate
    print("\nIMPLICATION gate performance after training:")
    implication_gate.evaluate(training_data=implication_data)
    implication_gate.plot_training_history()
    
    return implication_gate


#######################
# Part 3: Half Adder
#######################

def half_adder_example():
    """
    This function demonstrates how to use the HalfAdder class,
    which combines an XOR gate (for sum) and an AND gate (for carry).
    """
    print("\n" + "="*50)
    print("PART 3: HALF ADDER")
    print("="*50)
    
    # Create a half adder
    half_adder = HalfAdder(seed=SEED)
    
    # Train both gates of the half adder
    print("\nTraining the half adder (XOR for sum, AND for carry)...")
    sum_history, carry_history = half_adder.train(epochs=5000, learning_rate=0.1, verbose=VERBOSE)
    
    # Evaluate the half adder
    print("\nHalf adder performance after training:")
    half_adder.evaluate()
    
    # Visualize training progress for both gates
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sum_history)
    plt.title("Sum Gate (XOR) Training")
    plt.xlabel("Logged Epochs")
    plt.ylabel("Cost (MSE)")
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(carry_history)
    plt.title("Carry Gate (AND) Training")
    plt.xlabel("Logged Epochs")
    plt.ylabel("Cost (MSE)")
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate computation with the half adder
    print("\nComputing binary addition with the half adder:")
    print("-" * 40)
    
    for i in range(2):
        for j in range(2):
            sum_bit, carry_bit = half_adder.compute(i, j)
            binary_sum = carry_bit * 2 + sum_bit  # Convert to decimal
            print(f"{i} + {j} = {binary_sum} (Sum: {sum_bit}, Carry: {carry_bit})")
    
    return half_adder


#######################
# Part 4: Full Adder
#######################

def full_adder_example():
    """
    This function demonstrates how to use the FullAdder class,
    which combines two half adders and an OR gate to create a full adder.
    """
    print("\n" + "="*50)
    print("PART 4: FULL ADDER")
    print("="*50)
    
    # Create a full adder
    full_adder = FullAdder(seed=SEED)
    
    # Train all components of the full adder
    print("\nTraining the full adder...")
    full_adder.train(epochs=5000, learning_rate=0.1, verbose=VERBOSE)
    
    # Evaluate the full adder
    print("\nFull adder performance after training:")
    full_adder.evaluate()
    
    # Demonstrate computation with the full adder for all input combinations
    print("\nComputing binary addition with carry-in using the full adder:")
    print("-" * 60)
    
    print("| A | B | Cin | Sum | Cout | Decimal Representation |")
    print("|---|---|-----|-----|------|------------------------|")
    
    for a in range(2):
        for b in range(2):
            for cin in range(2):
                sum_bit, cout_bit = full_adder.compute(a, b, cin)
                decimal_sum = a + b + cin
                print(f"| {a} | {b} | {cin}   | {sum_bit}   | {cout_bit}    | {a} + {b} + {cin} = {decimal_sum}            |")
    
    return full_adder


#######################
# Part 5: Building a Binary Adder
#######################

def binary_adder_example():
    """
    This function demonstrates how to use multiple full adders
    to create a 4-bit binary adder.
    """
    print("\n" + "="*50)
    print("PART 5: 4-BIT BINARY ADDER")
    print("="*50)
    
    # Create and train four full adders (one for each bit)
    print("\nCreating and training 4 full adders for a 4-bit binary adder...")
    fa_list = []
    
    for i in range(4):
        print(f"\nTraining full adder {i+1}/4...")
        fa = FullAdder(seed=SEED + i)
        fa.train(epochs=5000, learning_rate=0.1, verbose=False)
        fa_list.append(fa)
    
    # Function to compute binary addition using the 4 full adders
    def add_4bit(a, b):
        """Add two 4-bit binary numbers."""
        # Convert decimal to 4-bit binary arrays
        a_bits = [(a >> i) & 1 for i in range(4)]
        b_bits = [(b >> i) & 1 for i in range(4)]
        
        # Initialize carry-in for the first adder
        carry = 0
        result_bits = []
        
        # Process each bit position using the corresponding full adder
        for i in range(4):
            sum_bit, carry = fa_list[i].compute(a_bits[i], b_bits[i], carry)
            result_bits.append(sum_bit)
    
        # Include overflow bit (5th bit) if needed
        result_bits.append(carry)
        
        # Convert binary result back to decimal
        decimal_result = 0
        for i, bit in enumerate(result_bits):
            decimal_result += bit * (2 ** i)
            
        # Return binary representation and decimal value
        return result_bits, decimal_result
    
    # Test the 4-bit adder with some examples
    print("\nTesting the 4-bit binary adder:")
    print("-" * 70)
    print("| Decimal A | Decimal B | Binary A  | Binary B  | Binary Sum  | Decimal Sum |")
    print("|-----------|-----------|-----------|-----------|-------------|-------------|")
    
    # Test with various input pairs
    test_cases = [(3, 5), (7, 8), (10, 5), (15, 15)]
    
    for a, b in test_cases:
        # Convert to binary strings for display
        a_bin = format(a, '04b')
        b_bin = format(b, '04b')
        
        # Compute using our neural adder
        result_bits, decimal_sum = add_4bit(a, b)
        
        # Format result for display (reversing bits for correct display)
        result_binary = ''.join([str(bit) for bit in result_bits[::-1]])
        
        print(f"| {a:9d} | {b:9d} | {a_bin:9s} | {b_bin:9s} | {result_binary:11s} | {decimal_sum:11d} |")
    
    print("\nNote: The binary adder can handle numbers up to 15 (4 bits).")
    print("If the sum exceeds 15, the 5th bit (leftmost) represents the overflow.")
    
    return fa_list


#######################
# Main function
#######################

def main():
    """
    Main function to run all examples sequentially.
    """  
    # Record the start time
    start_time = time.time()
    
    # Run all examples
    basic_gates = basic_logic_gates_example()
    custom_gate = custom_logic_gate_example()
    half_adder = half_adder_example()
    full_adder = full_adder_example()
    binary_adder = binary_adder_example()
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    
    print("\n" + "*"*70)
    print("* SUMMARY OF NEURAL LOGIC GATES CAPABILITIES:".ljust(69) + "*")
    print("*"*70)
    print("* 1. Basic logic gates (AND, OR, XOR, etc.)".ljust(69) + "*")
    print("* 2. Custom logic gates with user-defined truth tables".ljust(69) + "*")
    print("* 3. Half adder (combining XOR and AND gates)".ljust(69) + "*") 
    print("* 4. Full adder (combining half adders and OR gate)".ljust(69) + "*")
    print("* 5. Multi-bit binary adders".ljust(69) + "*")
    print("* 7. Training visualization".ljust(69) + "*")
    print("*"*70)
    
    print("\nThank you for exploring the NeuralGates!")
    print("seeyaa :)")


if __name__ == "__main__":
    main()