import numpy as np  # 导入 numpy 库并别名为 np；numpy 提供高性能的多维数组对象 ndarray
                   # 以及向量化数学运算，是强化学习中处理状态值函数、策略矩阵的核心工具
import random       # 导入标准库 random；提供伪随机数生成、序列随机排列等功能，
                   # 此处主要使用 random.seed() 固定种子 和 random.shuffle() 随机打乱列表


class GridWorld_v1(object):
    """
    网格世界环境（第一版，GridWorld v1）。

    【环境描述】
    构建一个 rows × columns 的有限二维网格（Finite MDP），每个格子对应一个离散状态。
    格子分三类：
        - 普通格子（奖励 = 0）：智能体可自由通行。
        - 目标区域（奖励 = score）：智能体到达后获得正奖励，用 ✅ 表示。
        - 禁止区域（奖励 = forbiddenAreaScore）：到达后获得负奖励（惩罚），用 🚫 表示。

    【设计目的】
    本版本专为"确定性环境"设计（即动作执行结果唯一确定，不含随机性），
    可直接用于求解以下两种经典 DP 算法的贝尔曼方程：
        - 价值迭代（Value Iteration）：通过反复更新状态价值函数 V(s) 至收敛
        - 策略迭代（Policy Iteration）：交替执行策略评估和策略改进

    【状态编号规则】
    第 i 行第 j 列的格子，其状态编号 s = i * columns + j
    例如 4×5 网格：左上角为 s=0，右上角为 s=4，左下角为 s=15，右下角为 s=19

    【动作空间（5 个离散动作）】
        A0: 向上移动    → 行坐标 -1，列坐标不变
        A1: 向右移动    → 行坐标不变，列坐标 +1
        A2: 向下移动    → 行坐标 +1，列坐标不变
        A3: 向左移动    → 行坐标不变，列坐标 -1
        A4: 原地不动    → 行列坐标均不变

    【类属性说明】
        stateMap (list[list[int]]): 形状为 (rows, columns) 的二维列表。
            stateMap[i][j] 存储第 i 行第 j 列格子的唯一状态编号（整数）。
            编号公式：stateMap[i][j] = i * columns + j。
            示例（3×3）：[[0,1,2],[3,4,5],[6,7,8]]

        scoreMap (np.ndarray): 形状为 (rows, columns) 的二维数组，dtype 为 float64。
            scoreMap[i][j] 存储第 i 行第 j 列格子对应的即时奖励 r(s)。
            取值只有三种：0（普通格子）、score（目标区域）、forbiddenAreaScore（禁止区域）。

        score (int | float): 目标区域的即时奖励值，对应强化学习中正奖励信号 r > 0。

        forbiddenAreaScore (int | float): 禁止区域的即时惩罚值，
            对应负奖励信号 r < 0，引导智能体规避危险区域。
    """

    # 类级别默认值（未实例化时的占位，实例化后会被 __init__ 覆盖为实例属性）
    stateMap = None          # 状态编号映射表，初始为 None；__init__ 后变为 list[list[int]]，形状 (rows, columns)
    scoreMap = None          # 奖励值映射表，初始为 None；__init__ 后变为 np.ndarray，形状 (rows, columns)
    score = 0                # 目标区域奖励默认值，类型 int；实例化时由构造参数 score 覆盖
    forbiddenAreaScore = 0   # 禁止区域惩罚默认值，类型 int；实例化时由构造参数 forbiddenAreaScore 覆盖

    # ================================================================== #
    # 一、初始化
    # ================================================================== #
    def __init__(self, rows=4, columns=5, forbiddenAreaNums=3, targetNums=1,
                 seed=-1, score=1, forbiddenAreaScore=-1, desc=None):
        """
        初始化网格世界，根据是否传入 desc 参数选择两种构建模式之一。

        【模式一：固定模式（desc 不为 None）】
            按照 desc 字符串列表精确描述网格布局，适用于需要可复现的固定实验场景。
            desc 示例：["T..#", "....", ".#..", "...."]
                'T' → 目标区域（score）
                '#' → 禁止区域（forbiddenAreaScore）
                其他字符（如 '.'）→ 普通格子（奖励 = 0）

        【模式二：随机模式（desc 为 None，默认）】
            按 rows、columns 创建网格，用 random.shuffle 随机确定
            forbiddenAreaNums 个禁止区域和 targetNums 个目标区域的位置。
            可通过 seed 固定随机结果，以保证实验可复现。

        参数：
            rows (int): 网格行数，随机模式使用，默认 4。取值建议 ≥ 1。
            columns (int): 网格列数，随机模式使用，默认 5。取值建议 ≥ 1。
            forbiddenAreaNums (int): 禁止区域（🚫）数量，随机模式使用，默认 3。
                应满足 forbiddenAreaNums + targetNums < rows * columns，否则格子不够分配。
            targetNums (int): 目标区域（✅）数量，随机模式使用，默认 1。
            seed (int): 随机数种子，传入任意整数可复现结果；传入 -1 则不固定种子，默认 -1。
            score (int | float): 目标区域的即时奖励值，对应 R(s) 中的正值，默认 1。
            forbiddenAreaScore (int | float): 禁止区域的即时惩罚值，对应 R(s) 中的负值，默认 -1。
            desc (list[str] | None): 固定模式的网格描述，每个字符串为一行网格；
                为 None 时使用随机模式，默认 None。

        返回：
            None（Python 构造函数约定不返回值）
        """
        # 将目标奖励值保存为实例属性，后续 scoreMap 填充和 show/showPolicy 均依赖此值
        # 类型：int 或 float，例如 score=1 表示到达目标格子获得 +1 奖励
        self.score = score

        # 将禁止区域惩罚值保存为实例属性，后续用作 scoreMap 的填充值和字典键
        # 类型：int 或 float，例如 forbiddenAreaScore=-1 表示进入禁止区域获得 -1 惩罚
        self.forbiddenAreaScore = forbiddenAreaScore

        # ---- 1.1 固定模式：根据字符串列表 desc 逐字符解析，构建 scoreMap ----
        if desc is not None:
            # len(desc) 返回列表元素个数，即行字符串的条数，等于网格行数
            # 类型：int，例如 desc=["T.", ".."] 则 self.rows = 2
            self.rows = len(desc)

            # len(desc[0]) 取第一行字符串的字符数，即网格列数
            # 假设每行字符数相同（调用方保证）；类型：int，例如 desc[0]="T." 则 self.columns = 2
            self.columns = len(desc[0])

            # 临时嵌套列表，外层对应行，内层对应列，最终转为 scoreMap
            # 初始为空列表，类型：list[list[int | float]]
            l = []

            # i：当前行的行索引，类型 int，范围 [0, self.rows)
            # 每次循环处理 desc 中第 i 行字符串 desc[i]
            for i in range(self.rows):
                # 初始化当前行的奖励列表，长度将在内层循环后达到 self.columns
                # 类型：list[int | float]
                tmp = []

                # j：当前列的列索引，类型 int，范围 [0, self.columns)
                # desc[i][j]：第 i 行第 j 列的字符，类型 str（单字符）
                for j in range(self.columns):
                    # 三元表达式链式判断字符类型并映射到对应奖励值：
                    #   desc[i][j] == '#' → 禁止区域，奖励为 forbiddenAreaScore（如 -1）
                    #   desc[i][j] == 'T' → 目标区域，奖励为 score（如 1）
                    #   其他字符           → 普通格子，奖励为 0
                    # append 将该值追加到 tmp 末尾
                    tmp.append(
                        forbiddenAreaScore if desc[i][j] == '#'
                        else score if desc[i][j] == 'T'
                        else 0
                    )

                # 将当前行奖励列表 tmp（长度 self.columns）追加到嵌套列表 l
                # l 追加后长度比之前多 1，类型：list[list[int | float]]
                l.append(tmp)

            # np.array(l) 将嵌套列表转换为二维 ndarray
            # 形状：(self.rows, self.columns)，dtype 由值自动推断（一般为 float64 或 int64）
            # 例如 l=[[0,1],[-1,0]] → array([[0.,1.],[-1.,0.]])
            self.scoreMap = np.array(l)

            # 列表推导式生成二维状态编号表，stateMap[i][j] = i * self.columns + j
            # 外层列表推导遍历行（i），内层列表推导遍历列（j）
            # 结果类型：list[list[int]]，形状逻辑上为 (self.rows, self.columns)
            # 示例（2×3）：[[0,1,2],[3,4,5]]
            self.stateMap = [
                [i * self.columns + j for j in range(self.columns)]
                for i in range(self.rows)
            ]

            # 固定模式构建完毕，提前 return，跳过下方随机模式的所有代码
            return

        # ---- 1.2 随机模式：根据 rows/columns/seed 随机生成网格布局 ----

        # 保存行数到实例属性，类型 int；后续所有行遍历均使用 self.rows
        self.rows = rows

        # 保存列数到实例属性，类型 int；后续列遍历、坐标转换均使用 self.columns
        self.columns = columns

        # 保存禁止区域数量，类型 int；用于后续 for 循环设置惩罚格子
        self.forbiddenAreaNums = forbiddenAreaNums

        # 保存目标区域数量，类型 int；用于后续 for 循环设置奖励格子
        self.targetNums = targetNums

        # 保存随机种子，类型 int；seed=-1 时 random.seed(-1) 依然会设置种子，
        # 若需要"每次不同"应传 None，但此处设计约定 -1 为"不固定"的语义标记
        self.seed = seed

        # 设置 Python 标准库 random 模块的全局随机种子
        # 作用：固定后续所有 random 操作的随机序列，保证实验在相同 seed 下可复现
        # 例如 seed=42 时，每次运行 shuffle 结果完全一致
        random.seed(self.seed)

        # 生成从 0 到 rows*columns-1 的连续整数列表，长度 = rows*columns
        # 每个整数对应一个格子的"一维索引"（展平坐标）
        # 类型：list[int]，例如 rows=2, columns=3 → [0,1,2,3,4,5]
        l = [i for i in range(self.rows * self.columns)]

        # random.shuffle(l) 原地随机打乱 l，打乱后 l 中整数顺序随机
        # 打乱的目的：将前 forbiddenAreaNums 个元素作为禁止格子的一维索引，
        # 紧接着 targetNums 个元素作为目标格子的一维索引，从而实现随机布局
        # 无返回值，直接修改 l 本身
        random.shuffle(l)

        # 初始化一维奖励列表，长度 rows*columns，所有格子初始奖励为 0（普通格子）
        # 类型：list[int]，例如 rows*columns=6 → [0,0,0,0,0,0]
        self.g = [0 for i in range(self.rows * self.columns)]

        # i：循环计数器，类型 int，范围 [0, forbiddenAreaNums)
        # l[i]：打乱后列表的第 i 个元素，即第 i 个被选中的禁止区域在一维展平中的位置索引
        for i in range(forbiddenAreaNums):
            # 将该位置的奖励值设置为 forbiddenAreaScore（如 -1），标记为禁止区域
            self.g[l[i]] = forbiddenAreaScore

        # i：循环计数器，类型 int，范围 [0, targetNums)
        # l[forbiddenAreaNums + i]：跳过前 forbiddenAreaNums 个（已用于禁止区域），
        # 取紧接其后的第 i 个元素作为目标区域的一维位置索引
        for i in range(targetNums):
            # 将该位置的奖励值设置为 score（如 1），标记为目标区域
            self.g[l[forbiddenAreaNums + i]] = score

        # np.array(self.g) 将一维列表 self.g 转为一维 ndarray，长度 rows*columns
        # .reshape(rows, columns) 将一维数组重塑为二维 ndarray，形状 (rows, columns)
        # 等价于将展平的格子奖励恢复为网格的二维布局
        # dtype 自动推断，通常为 int64 或 float64
        self.scoreMap = np.array(self.g).reshape(rows, columns)

        # 与固定模式完全相同的状态编号表生成逻辑
        # stateMap[i][j] = i * self.columns + j，唯一标识每个格子
        # 类型：list[list[int]]，形状逻辑上为 (self.rows, self.columns)
        self.stateMap = [
            [i * self.columns + j for j in range(self.columns)]
            for i in range(self.rows)
        ]

    # ================================================================== #
    # 二、网格可视化
    # ================================================================== #
    def show(self):
        """
        在终端以 Emoji 字符可视化当前网格世界的地图布局。

        遍历 scoreMap 的每个元素，根据奖励值查表映射为对应 Emoji，
        逐行拼接并打印，直观展示网格中各类区域的空间分布。

        符号含义：
            ⬜️ → 普通可通行格子，即时奖励 = 0
            🚫 → 禁止区域，即时奖励 = forbiddenAreaScore（负值）
            ✅ → 目标区域，即时奖励 = score（正值）

        参数：
            无

        返回：
            None（输出直接打印到 stdout，无返回值）

        使用示例：
            env = GridWorld_v1(rows=3, columns=3)
            env.show()
            # 输出类似：
            # ⬜️🚫⬜️
            # ⬜️⬜️✅
            # 🚫⬜️⬜️
        """
        # i：当前行的行索引，类型 int，范围 [0, self.rows)
        # 每次外层循环对应网格的一行
        for i in range(self.rows):
            # 初始化当前行的输出字符串，类型 str，初始为空字符串
            # 内层循环会依次拼接 self.columns 个 Emoji 字符
            s = ""

            # j：当前列的列索引，类型 int，范围 [0, self.columns)
            # self.scoreMap[i][j]：第 i 行第 j 列格子的奖励值，类型 np.int64 或 np.float64
            for j in range(self.columns):

                # 构建"奖励值 → Emoji"的映射字典
                # 键：奖励数值（int/float，共三种取值），值：对应的 Emoji 字符串（str）
                # 注意：键使用 self.forbiddenAreaScore 和 self.score 而非硬编码，
                # 支持任意自定义奖励值
                tmp = {0: "⬜️", self.forbiddenAreaScore: "🚫", self.score: "✅"}

                # 用 self.scoreMap[i][j] 作为键查询 tmp 字典，得到对应 Emoji 字符串
                # 将结果追加到当前行字符串 s 末尾，类型 str
                s = s + tmp[self.scoreMap[i][j]]

            # 内层循环结束，s 已拼接当前行所有 self.columns 个 Emoji
            # print(s) 将当前行输出到终端，自动换行
            print(s)

    # ================================================================== #
    # 三、状态转移与奖励查询
    # ================================================================== #
    def getScore(self, nowState, action):
        """
        环境核心接口：给定当前状态和动作，返回即时奖励与下一状态。

        对应强化学习中的环境转移函数 T(s, a) → (r, s')，
        在确定性环境下，每个 (s, a) 对唯一对应一个 (r, s')。

        【边界处理（撞墙）】
        当动作使智能体试图移出网格边界时（如在最左列执行向左动作），
        视为"撞墙"：智能体位置不变（s' = s），返回固定惩罚 r = -1。
        注意：撞墙惩罚 -1 是硬编码值，与 forbiddenAreaScore 无关。

        参数：
            nowState (int): 当前状态编号，范围 [0, rows*columns)。
                编号规则：nowState = 行索引 × columns + 列索引。
                例如 4×5 网格中，状态 7 对应第 1 行第 2 列（0-indexed）。
            action (int): 动作编号，范围 [0, 4]，含义：
                0 → 上（行-1），1 → 右（列+1），2 → 下（行+1），
                3 → 左（列-1），4 → 不动（行列不变）。

        返回：
            tuple (reward, nextState)：
                reward (int | float | np.int64 | np.float64):
                    即时奖励值。正常转移时为 nextState 对应格子的 scoreMap 值；
                    撞墙时固定返回整数 -1。
                nextState (int):
                    下一状态的编号。正常转移时为目标格子的 stateMap 值；
                    撞墙时返回 nowState 本身（保持不动）。

        使用示例：
            env = GridWorld_v1(rows=4, columns=5)
            reward, next_s = env.getScore(7, 0)  # 状态7执行动作0（向上）
        """
        # 从状态编号反算行坐标：整除列数得到所在行（行索引从 0 开始）
        # 类型：int；例如 nowState=7, columns=5 → nowx = 7//5 = 1（第1行）
        nowx = nowState // self.columns

        # 从状态编号反算列坐标：对列数取余得到所在列（列索引从 0 开始）
        # 类型：int；例如 nowState=7, columns=5 → nowy = 7%5 = 2（第2列）
        nowy = nowState % self.columns

        # nowState 超出合法范围（不应出现，属于调用方错误）
        # 此处仅打印警告，不抛出异常，便于调试时定位问题
        if nowx < 0 or nowy < 0 or nowx >= self.rows or nowy >= self.columns:
            # f-string 格式化输出非法坐标，类型 str
            print(f"coordinate error: ({nowx},{nowy})")

        # 动作编号不在 [0, 4] 的合法范围内（不应出现，属于调用方错误）
        if action < 0 or action >= 5:
            # 同样仅打印警告，不抛出异常
            print(f"action error: ({action})")

        # 定义五个动作对应的坐标偏移量 (Δrow, Δcol)
        # 索引 0~4 分别对应动作 上/右/下/左/不动
        # 类型：list[tuple[int, int]]，共 5 个元素
        # 坐标系约定：行向下为正（row+1 = 向下），列向右为正（col+1 = 向右）
        actionList = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

        # 计算执行 action 后到达的目标格子行坐标
        # actionList[action][0] 取第 action 个偏移量的 Δrow 分量，类型 int
        # tmpx = nowx + Δrow，类型 int
        tmpx = nowx + actionList[action][0]

        # 计算执行 action 后到达的目标格子列坐标
        # actionList[action][1] 取第 action 个偏移量的 Δcol 分量，类型 int
        # tmpy = nowy + Δcol，类型 int
        tmpy = nowy + actionList[action][1]

        # 目标坐标 (tmpx, tmpy) 超出网格边界，即"撞墙"
        if tmpx < 0 or tmpy < 0 or tmpx >= self.rows or tmpy >= self.columns:
            # 行为：智能体原地不动（s' = nowState），获得固定撞墙惩罚 r = -1
            # 返回：tuple(int, int)，第一个 -1 为惩罚，第二个 nowState 为原状态编号
            return -1, nowState

        # 目标坐标合法，从 scoreMap 中读取目标格子的即时奖励
        # self.scoreMap[tmpx][tmpy]：类型 np.int64 或 np.float64，
        #   值为 0（普通）、score（目标）或 forbiddenAreaScore（禁止区域）
        # self.stateMap[tmpx][tmpy]：目标格子的状态编号，类型 int，
        #   值 = tmpx * self.columns + tmpy
        return self.scoreMap[tmpx][tmpy], self.stateMap[tmpx][tmpy]

    # ================================================================== #
    # 四、策略可视化
    # ================================================================== #
    def showPolicy(self, policy):
        """
        在终端以 Emoji 箭头可视化每个状态下当前策略所选择的动作方向。

        遍历所有状态，根据格子类型（普通/目标/禁止）和策略动作编号，
        选择对应 Emoji 拼接输出，逐行打印形成策略地图。

        【视觉编码规则】
            ✅          → 目标区域，不显示方向（无需决策）
            ⬆️ ➡️ ⬇️ ⬅️ 🔄 → 普通格子，显示常规方向箭头
            ⏫️ ⏩️ ⏬ ⏪ 🔄  → 禁止区域，显示加速双箭头（提示危险，需快速离开）

        参数：
            policy (list[int] | np.ndarray): 长度为 rows*columns 的一维策略向量。
                policy[s] 表示在状态 s 下选择的动作编号（int），范围 [0, 4]。
                动作映射：0=上，1=右，2=下，3=左，4=不动/原地。
                例如 policy=[0,1,2,3,4,...] 表示状态0选上，状态1选右，以此类推。

        返回：
            None（输出直接打印到 stdout，无返回值）

        使用示例：
            policy = [0] * (env.rows * env.columns)  # 所有状态均选择"向上"
            env.showPolicy(policy)
        """
        # 将行数存入局部变量，类型 int，避免在循环中重复通过 self 查找属性（轻微优化）
        rows = self.rows

        # 将列数存入局部变量，类型 int，后续坐标换算和换行判断均使用此变量
        columns = self.columns

        # 当前行的输出字符串，类型 str，初始为空；每拼满一行后打印并重置
        s = ""

        # i：状态编号，类型 int，范围 [0, rows*columns)
        # 由于 stateMap[row][col] = row*columns+col，i 直接等于状态编号
        for i in range(self.rows * self.columns):
            # 由状态编号 i 反算行坐标：整除列数，类型 int
            # 例如 i=7, columns=5 → nowx=1（第1行）
            nowx = i // columns

            # 由状态编号 i 反算列坐标：对列数取余，类型 int
            # 例如 i=7, columns=5 → nowy=2（第2列）
            nowy = i % columns

            # 当前格子是目标区域（奖励等于 self.score）
            # 目标区域不需要展示策略方向，固定显示 ✅
            if self.scoreMap[nowx][nowy] == self.score:
                # 将 "✅" 字符串追加到当前行字符串 s 末尾
                s = s + "✅"

            # 当前格子是普通可通行区域（奖励为 0）
            if self.scoreMap[nowx][nowy] == 0:
                # 构建"动作编号 → 普通箭头 Emoji"的映射字典
                # 键：动作编号（int，0~4），值：对应 Emoji 字符串（str）
                # 0=上⬆️  1=右➡️  2=下⬇️  3=左⬅️  4=不动🔄
                tmp = {0: "⬆️", 1: "➡️", 2: "⬇️", 3: "⬅️", 4: "🔄"}

                # policy[i] 取状态 i 对应的动作编号，类型 int，范围 [0, 4]
                # tmp[policy[i]] 查表得到对应 Emoji，追加到行字符串
                s = s + tmp[policy[i]]

            # 当前格子是禁止区域（奖励等于 self.forbiddenAreaScore，负值）
            if self.scoreMap[nowx][nowy] == self.forbiddenAreaScore:
                # 构建"动作编号 → 加速双箭头 Emoji"的映射字典
                # 双箭头视觉上更强烈，暗示智能体应快速离开危险区域
                # 0=上⏫️  1=右⏩️  2=下⏬  3=左⏪  4=原地🔄
                tmp = {0: "⏫️", 1: "⏩️", 2: "⏬", 3: "⏪", 4: "🔄"}

                # policy[i] 取当前禁止区域状态的策略动作编号，类型 int
                # 查表得到对应双箭头 Emoji，追加到行字符串
                s = s + tmp[policy[i]]

            # 当列索引等于最后一列索引（columns-1）时，当前行所有格子已处理完毕
            if nowy == columns - 1:
                # 打印拼接好的整行 Emoji 字符串 s，print 自动添加换行符
                print(s)

                # 重置行字符串为空，准备拼接下一行的 Emoji 序列
                s = ""
