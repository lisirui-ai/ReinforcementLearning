import numpy as np
import random


class GridWorld_v2(object):
    """
    网格世界环境（随机策略版）。

    与 v1 版本的两点区别：
    1. v1 针对确定性（deterministic）策略；v2 针对随机（stochastic）策略，
       策略矩阵 shape==(状态数, 5)，每行为一个概率分布，表示在该状态下选择各动作的概率，行和为 1。
    2. 新增 getTrajectoryScore 方法，可按照给定策略进行多步轨迹采样。

    打印策略时，显示每个状态下概率最大的动作。

    动作编号约定：
        0: 向上（↑），1: 向右（→），2: 向下（↓），3: 向左（←），4: 原地不动

    属性：
        stateMap          (list[list[int]]): rows×columns 的状态编号矩阵，stateMap[i][j] = i*columns+j
        scoreMap          (np.ndarray):      rows×columns 的即时奖励矩阵，shape=(rows, columns)
        score             (float):           目标区域（T）的即时奖励值
        forbiddenAreaScore(float):           禁止区域（#）的即时奖励值，通常为负数
    """

    stateMap = None           # rows×columns 的状态编号矩阵，list[list[int]]
    scoreMap = None           # rows×columns 的即时奖励矩阵，np.ndarray，shape=(rows, columns)
    score = 0                 # 目标区域的即时奖励值，float，默认 0
    forbiddenAreaScore = 0    # 禁止区域的即时奖励值，float，默认 0

    def __init__(self, rows=4, columns=5, forbiddenAreaNums=3, targetNums=1,
                 seed=-1, score=1, forbiddenAreaScore=-1, desc=None):
        """
        初始化网格世界，支持描述字符串模式和随机生成模式两种方式。

        描述字符串模式（desc 不为 None）：
            用字符矩阵指定布局，'.' 表示普通格，'#' 表示禁止区，'T' 表示目标区。

        随机生成模式（desc 为 None）：
            根据 rows/columns/forbiddenAreaNums/targetNums/seed 随机布置网格。

        参数：
            rows               (int):        网格行数，默认 4
            columns            (int):        网格列数，默认 5
            forbiddenAreaNums  (int):        随机模式下禁止区域的数量，默认 3
            targetNums         (int):        随机模式下目标区域的数量，默认 1
            seed               (int):        随机数种子，-1 表示不固定，默认 -1
            score              (float):      目标区域的即时奖励，默认 1
            forbiddenAreaScore (float):      禁止区域的即时奖励，默认 -1
            desc               (list[str] | None): 描述字符串列表，None 表示随机模式，默认 None
        """
        self.score = score                            # 保存目标区域奖励值，float
        self.forbiddenAreaScore = forbiddenAreaScore  # 保存禁止区域奖励值，float

        if desc is not None:
            self.rows = len(desc)                     # 从描述字符串推断行数，int
            self.columns = len(desc[0])               # 从描述字符串第一行推断列数，int
            l = []                                    # 逐行构建奖励矩阵的临时列表，list[list[float]]
            for i in range(self.rows):
                tmp = []                              # 当前行的奖励值列表，list[float]
                for j in range(self.columns):
                    # 字符映射规则：'#'→forbiddenAreaScore，'T'→score，'.'→0
                    tmp.append(forbiddenAreaScore if desc[i][j] == '#' else score if desc[i][j] == 'T' else 0)
                l.append(tmp)                         # 将当前行追加到总列表
            self.scoreMap = np.array(l)               # 转为 np.ndarray，shape=(rows, columns)，dtype=float
            # 生成状态编号矩阵，stateMap[i][j] = i*columns+j，list[list[int]]，shape=(rows, columns)
            self.stateMap = [[i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]
            return                                    # 描述字符串模式初始化完毕，直接返回

        # ---------- 随机模式 ----------
        self.rows = rows                              # 保存行数，int
        self.columns = columns                        # 保存列数，int
        self.forbiddenAreaNums = forbiddenAreaNums    # 保存禁止区域数量，int
        self.targetNums = targetNums                  # 保存目标区域数量，int
        self.seed = seed                              # 保存随机种子，int

        random.seed(self.seed)                        # 固定随机种子，保证结果可复现
        l = [i for i in range(self.rows * self.columns)]  # 生成 0~rows*columns-1 的状态编号列表，list[int]
        random.shuffle(l)                             # 原地随机打乱列表，用于随机分配禁止/目标区域位置
        self.g = [0 for _ in range(self.rows * self.columns)]  # 初始化奖励列表，全为 0，list[float]

        for i in range(forbiddenAreaNums):
            self.g[l[i]] = forbiddenAreaScore        # 将前 forbiddenAreaNums 个随机位置设为禁止区奖励

        for i in range(targetNums):
            self.g[l[forbiddenAreaNums + i]] = score # 将随后 targetNums 个随机位置设为目标区奖励

        # 将一维奖励列表转为 np.ndarray 并调整为 (rows, columns) 形状，dtype=float
        self.scoreMap = np.array(self.g).reshape(rows, columns)
        # 生成状态编号矩阵，list[list[int]]，shape=(rows, columns)
        self.stateMap = [[i * self.columns + j for j in range(self.columns)] for i in range(self.rows)]

    def show(self):
        """
        以 emoji 可视化打印网格世界布局。

        普通格显示 ⬜️，禁止区显示 🚫，目标区显示 ✅。
        无参数，无返回值，直接打印到控制台。
        """
        for i in range(self.rows):
            s = ""                                          # 当前行的 emoji 字符串，逐格拼接，str
            for j in range(self.columns):
                # 建立奖励值到 emoji 的映射字典，dict
                tmp = {0: "⬜️", self.forbiddenAreaScore: "🚫", self.score: "✅"}
                s = s + tmp[self.scoreMap[i][j]]            # 根据当前格奖励值选择对应 emoji 并拼接
            print(s)                                        # 打印当前行的 emoji 字符串

    def getScore(self, nowState, action):
        """
        计算从当前状态执行指定动作后获得的即时奖励及转移到的下一状态。

        边界处理：若动作导致坐标越界，则保持原地不动，即时奖励返回 -1。

        参数：
            nowState (int): 当前状态编号，取值范围 [0, rows*columns)
            action   (int): 动作编号，0=上，1=右，2=下，3=左，4=不动

        返回：
            score     (float): 进入下一状态的即时奖励
            nextState (int):   执行动作后的下一状态编号
        """
        nowx = nowState // self.columns  # 当前状态的行坐标（行索引），int
        nowy = nowState % self.columns   # 当前状态的列坐标（列索引），int

        if nowx < 0 or nowy < 0 or nowx >= self.rows or nowy >= self.columns:
            print(f"coordinate error: ({nowx},{nowy})")  # 坐标越界时打印错误提示

        if action < 0 or action >= 5:
            print(f"action error: ({action})")           # 动作非法时打印错误提示

        # 动作对应的行列偏移量列表：上(-1,0)，右(0,1)，下(1,0)，左(0,-1)，不动(0,0)
        actionList = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        tmpx = nowx + actionList[action][0]  # 执行动作后的行坐标，int
        tmpy = nowy + actionList[action][1]  # 执行动作后的列坐标，int

        if tmpx < 0 or tmpy < 0 or tmpx >= self.rows or tmpy >= self.columns:
            return -1, nowState              # 越界时返回惩罚 -1 和原状态编号

        # 返回目标格的即时奖励（float）和对应状态编号（int）
        return self.scoreMap[tmpx][tmpy], self.stateMap[tmpx][tmpy]

    def getTrajectoryScore(self, nowState, action, policy, steps, stop_when_reach_target=False):
        """
        从给定起始状态和动作出发，按指定策略进行多步轨迹采样，并记录每步信息。

        参数：
            nowState               (int):         起始状态编号
            action                 (int):         起始动作编号
            policy                 (np.ndarray):  策略矩阵，shape=(rows*columns, 5)，
                                                  每行为动作概率分布，元素非负且行和为 1
            steps                  (int):         采样步数；返回列表长度为 steps+1（含第 0 步）
            stop_when_reach_target (bool):        是否在首次到达目标区域时提前终止，默认 False；
                                                  为 True 时 steps 自动扩大到 20000 以确保能到达目标

        返回：
            res (list[tuple]): 长度为 steps+1 的轨迹列表（stop_when_reach_target=True 时可能更短），
                               每个元素为五元组 (nowState, nowAction, score, nextState, nextAction)：
                                   nowState   (int):   当前步的状态编号
                                   nowAction  (int):   当前步的动作编号
                                   score      (float): 当前步的即时奖励
                                   nextState  (int):   当前步转移到的下一状态编号
                                   nextAction (int):   下一步将执行的动作编号（按策略采样）
        """
        res = []                    # 存储轨迹的结果列表，list[tuple]
        nextState = nowState        # 初始化下一状态为起始状态，int
        nextAction = action         # 初始化下一动作为起始动作，int

        if stop_when_reach_target:  # 若需在到达目标时提前终止，扩大最大步数以保证能抵达
            steps = 20000

        for i in range(steps + 1):  # 循环 steps+1 次，第 0 步也被记录
            nowState = nextState    # 将上一步的下一状态更新为当前状态，int
            nowAction = nextAction  # 将上一步选出的下一动作更新为当前动作，int

            # 执行动作，获取即时奖励（float）和下一状态编号（int）
            score, nextState = self.getScore(nowState, nowAction)
            # np.random.choice(range(5), size=1, replace=False, p=policy[nextState])：
            # 从候选动作 [0,1,2,3,4] 中按概率分布 policy[nextState]（np.ndarray，shape=(5,)，元素非负且和为1）
            # 不放回地随机采样 1 个动作——概率越大的动作被选中的可能性越高，但不保证一定选到概率最大的动作；
            # 返回 np.ndarray，shape=(1,)，dtype=int64，如 array([2])；
            # [0] 取出唯一元素，nextAction 为 np.int64，即下一步执行的动作编号，取值范围 [0, 4]
            nextAction = np.random.choice(range(5), size=1, replace=False, p=policy[nextState])[0]

            # 将当前步的完整信息打包为五元组并追加到结果列表
            res.append((nowState, nowAction, score, nextState, nextAction))

            if stop_when_reach_target:
                nowx = nowState // self.columns  # 当前状态的行坐标，int
                nowy = nowState % self.columns   # 当前状态的列坐标，int
                if self.scoreMap[nowx][nowy] == self.score:  # 已到达目标区域，提前返回
                    return res

        return res  # 返回完整轨迹列表，list[tuple]，长度为 steps+1

    def showPolicy(self, policy):
        """
        以 emoji 可视化策略，显示每个状态下概率最大的动作。

        普通格使用普通箭头（⬆️➡️⬇️⬅️🔄），禁止区使用加急双箭头（⏫️⏩️⏬⏪🔄），目标区显示 ✅。

        参数：
            policy (np.ndarray): 策略矩阵，shape=(rows*columns, 5)，每行为动作概率分布
        无返回值，直接打印到控制台。
        """
        rows = self.rows        # 网格行数，int
        columns = self.columns  # 网格列数，int
        s = ""                  # 当前行的 emoji 字符串，逐格拼接，str

        for i in range(self.rows * self.columns):
            nowx = i // columns  # 当前状态的行坐标，int
            nowy = i % columns   # 当前状态的列坐标，int

            if self.scoreMap[nowx][nowy] == self.score:
                s = s + "✅"     # 目标区域直接显示 ✅

            if self.scoreMap[nowx][nowy] == 0:
                # 普通格动作到 emoji 的映射：0=上 1=右 2=下 3=左 4=原地
                tmp = {0: "⬆️", 1: "➡️", 2: "⬇️", 3: "⬅️", 4: "🔄"}
                s = s + tmp[np.argmax(policy[i])]   # 取概率最大的动作索引并映射为 emoji

            if self.scoreMap[nowx][nowy] == self.forbiddenAreaScore:
                # 禁止格动作到加急双箭头 emoji 的映射：0=上 1=右 2=下 3=左 4=原地
                tmp = {0: "⏫️", 1: "⏩️", 2: "⏬", 3: "⏪", 4: "🔄"}
                s = s + tmp[np.argmax(policy[i])]   # 取概率最大的动作索引并映射为双箭头 emoji

            if nowy == columns - 1:
                print(s)  # 当前行最后一格处换行打印
                s = ""    # 重置当前行字符串，准备拼接下一行
