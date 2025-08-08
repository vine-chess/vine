import sys


def main():
    if len(sys.argv) != 4:
        print("usage: python3 concat_networks.py <value net> <policy net> <output>")
        sys.exit(1)

    value_net_path = sys.argv[1]
    policy_net_path = sys.argv[2]
    output_path = sys.argv[3]

    if value_net_path.endswith(".pn"):
        print("warning: value net has policy net extension")
    if policy_net_path.endswith(".vn"):
        print("warning: policy net has value net extension")

    with open(value_net_path, 'rb') as f:
        value_net_data = f.read()

    with open(policy_net_path, 'rb') as f:
        policy_net_data = f.read()

    value_net_size = len(value_net_data)
    padding_size = (64 - (value_net_size % 64)) % 64
    padding = b'\x00' * padding_size

    with open(output_path, 'wb') as combined:
        combined.write(value_net_data)
        combined.write(padding)
        combined.write(policy_net_data)


if __name__ == "__main__":
    main()
