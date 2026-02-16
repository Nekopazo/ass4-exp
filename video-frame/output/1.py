import os

def remove_conf_from_txt():
    for filename in os.listdir("."):
        if filename.endswith(".txt"):
            with open(filename, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                
                # 如果是6列（class + 4 box + conf）
                if len(parts) == 6:
                    parts = parts[:5]   # 删除最后一列
                
                new_lines.append(" ".join(parts) + "\n")

            with open(filename, "w") as f:
                f.writelines(new_lines)

    print("Done. All confidence values removed.")

if __name__ == "__main__":
    remove_conf_from_txt()
