with open("parameters2.txt", "w") as f:
    i = 0
    for model_name in ["dinov2_vitb14_reg"]:
        for lr in [0.00005, 0.0001]:
            for dropout_p in [0.1, 0.15, 0.25]:
                for label_smoothing in [0.1, 0.2]:
                    for lora, rank in [(0, 0), (1, 32), (1, 64)]:
                        for resize_max in [896, 1400]:
                            for pos_embed in ["pre-trained", "fine-tune"]:
                                if pos_embed == "fine-tune" and not lora:
                                    continue
                                i += 1
                                cmd = f"{model_name} {lr} {dropout_p} {label_smoothing} {lora} {rank} 224 {resize_max} {pos_embed}\n"
                                print(cmd)
                                f.writelines(cmd)
print(i)
