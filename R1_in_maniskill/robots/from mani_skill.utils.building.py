from mani_skill.utils.building.articulations import partnet_mobility
print(partnet_mobility.PARTNET_MOBILITY is None)
# 如果上面打印 True，说明 PARTNET_MOBILITY 没有加载到任何元数据
# 如果打印 False，再打印 model_data 的 keys 数量
if partnet_mobility.PARTNET_MOBILITY is not None:
    keys = list(partnet_mobility.PARTNET_MOBILITY["model_data"].keys())
    print("共加载到 PartNet Mobility 模型 ID：", len(keys))
    # 打印前 20 个 ID，检查是否出现你想要的
    print("前 20 个 ID：", keys[:20])
    print("包含 140c308db70cd2674da5feafe6f1c8fc 吗？", "140c308db70cd2674da5feafe6f1c8fc" in keys)
