# trsfl_system.py
import time
import random
import secrets
import hashlib
import hmac
from charm.toolbox.ecgroup import ECGroup, ZR, G
import sympy
from Crypto.Cipher import ChaCha20

# ==================== chacha20 ====================
def chacha20_prg(key: bytes, nonce: bytes, length: int) -> bytes:
    """生成 length 字节的伪随机序列，初始块计数器为 0。"""
    if len(key) != 32:
        raise ValueError("密钥必须为 32 字节")
    if len(nonce) != 12:
        raise ValueError("nonce 必须为 12 字节")
    if not 0 <= length <= (1 << 38):
        raise ValueError("输出长度超出允许范围")

    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.encrypt(bytes(length))

# ==================== Asmuth-Bloom 秘密共享 ====================
def _generate_prime(bits: int, lower_bound: int = None) -> int:
    if lower_bound is None:
        lower_bound = 1 << (bits - 1)
    else:
        lower_bound = max(lower_bound, 1 << (bits - 1))
    n = lower_bound if lower_bound % 2 == 1 else lower_bound + 1
    while True:
        if sympy.isprime(n):
            return n
        n += 2

def _generate_coprime_moduli(m0: int, n: int, bits: int, min_val: int = None) -> list:
    moduli = []
    current = min_val if min_val is not None else m0 + 1
    for _ in range(n):
        p = _generate_prime(bits, current)
        moduli.append(p)
        current = p + 1
    return moduli

def _condition_holds(m0: int, moduli: list, l: int) -> bool:
    n = len(moduli)
    left = 1
    for i in range(l):
        left *= moduli[i]
    right = m0
    for i in range(n - l + 1, n):
        right *= moduli[i]
    return left > right

def _crt_recover(pairs):
    M = 1
    for _, mi in pairs:
        M *= mi
    x = 0
    for si, mi in pairs:
        Mi = M // mi
        inv = pow(Mi, -1, mi)
        x += si * Mi * inv
    return x % M

def share_secret(secret: int, n: int, l: int, secret_bits: int = 500):
    """返回 (m0, moduli, shares)"""
    m0 = _generate_prime(secret_bits + 5, secret + 1)
    bits = (secret_bits // l) + 20
    while True:
        moduli = _generate_coprime_moduli(m0, n, bits, min_val=m0 + 1)
        if _condition_holds(m0, moduli, l):
            break
        bits += 10
    L = 1
    for i in range(l):
        L *= moduli[i]
    max_v = (L - secret - 1) // m0
    if max_v < 0:
        raise ValueError("秘密过大")
    v = random.randrange(0, max_v + 1)
    beta = secret + v * m0
    shares = [beta % mi for mi in moduli]
    return m0, moduli, shares

def reconstruct_secret(shares_with_moduli, m0):
    """shares_with_moduli: [(share, modulus), ...]"""
    if not shares_with_moduli:
        raise ValueError("至少需要一个碎片")
    beta = _crt_recover(shares_with_moduli)
    return beta % m0

# ==================== TA 类 ====================
class TA:
    def __init__(self, curve_id=714, lambda_param=256):
        self.group = ECGroup(curve_id)
        self.q = self.group.order()
        self.P = self.group.random(G)
        self.lambda_param = lambda_param
        self.OUT_BYTES = lambda_param // 8

        # 主密钥 msk = x
        self.x = self.group.random(ZR)
        self.P_pub = self.P ** self.x

        # 公共盐
        self.s = ''.join(secrets.choice('01') for _ in range(self.OUT_BYTES))

        self.pp = {
            "q": self.q, "P": self.P, "P_pub": self.P_pub, "s": self.s,
            "H1": self.H1, "H2": self.H2, "H3": self.H3, "H4": self.H4,
            "H5": self.H5, "H6": self.H6, "H7": self.H7, "H": self.H
        }

    def _hkdf(self, data: bytes, length: int, info: bytes) -> bytes:
        salt = b"TRS-FL"
        prk = hmac.new(salt, data, hashlib.sha256).digest()
        out, prev = b"", b""
        block = 32
        for i in range(1, (length + block - 1) // block + 1):
            prev = hmac.new(prk, prev + info + bytes([i]), hashlib.sha256).digest()
            out += prev
        return out[:length]

    def _hash_to_zq(self, data: bytes) -> ZR:
        seed = self._hkdf(data, self.OUT_BYTES, info=b"ZR-MAP")
        return self.group.hash(seed, ZR)

    def mask_vector(self, k_ij, PID_ij_t: bytes, Y_ij):
        """用轮密钥及会话上下文确定性生成 64 维 ZR 掩码。"""
        q = int(self.q)
        if q.bit_length() > 256:
            raise ValueError("轮密钥所在群的阶不能超过 256 位")
        key = int(k_ij).to_bytes(32, 'big')
        Y_bytes = self.group.serialize(Y_ij)
        context = (b"TRS-FL-MASK" + (64).to_bytes(4, 'big') +
                   q.to_bytes(32, 'big') + len(PID_ij_t).to_bytes(4, 'big') +
                   PID_ij_t + len(Y_bytes).to_bytes(4, 'big') + Y_bytes)
        # 每个轮密钥仅签密一次；AS 可从相同上下文重现 nonce 和掩码。
        nonce = hashlib.sha256(context).digest()[:12]
        stream = chacha20_prg(key, nonce, 64 * 32)
        return [self.group.init(ZR, int.from_bytes(stream[i:i + 32], 'big') % q)
                for i in range(0, len(stream), 32)]

    def _serialize_ciphertext(self, C_ij) -> bytes:
        if len(C_ij) != 64:
            raise ValueError("密文必须为 64 维向量")
        encoded = [self.group.serialize(value) for value in C_ij]
        return len(encoded).to_bytes(4, 'big') + b''.join(
            len(value).to_bytes(4, 'big') + value for value in encoded)

    def H1(self, sk_ij_puf: bytes, s: str) -> bytes:
        return self._hkdf(sk_ij_puf + s.encode(), self.OUT_BYTES, b"H1")

    def H2(self, Com_ij: bytes, X_ij, P_pub) -> bytes:
        data = Com_ij + self.group.serialize(X_ij) + self.group.serialize(P_pub)
        return self._hkdf(data, self.OUT_BYTES, b"H2")

    def H3(self, PID_ij: bytes, P_pub, X_ij, R_ij) -> ZR:
        data = PID_ij + self.group.serialize(P_pub) + self.group.serialize(X_ij) + self.group.serialize(R_ij)
        return self._hash_to_zq(data)

    def H4(self, nonce_ij_t: str, Com_ij: bytes) -> bytes:
        return self._hkdf(nonce_ij_t.encode() + Com_ij, self.OUT_BYTES, b"H4")

    def H5(self, PID_ij: bytes, t: int, Com_ij_t: bytes) -> bytes:
        data = PID_ij + t.to_bytes(self.OUT_BYTES, 'big') + Com_ij_t
        return self._hkdf(data, self.OUT_BYTES, b"H5")
    '''
    def H6(self, alpha_i: int, PID_ij_t: bytes, kappa_ij) -> ZR:
        data = alpha_i.to_bytes(self.OUT_BYTES, 'big') + PID_ij_t + self.group.serialize(kappa_ij)
        return self._hash_to_zq(data)
    '''
    def H6(self, alpha_i: int, PID_ij_t: bytes, kappa_ij) -> ZR:
    # 将大整数 alpha_i 转换为字节（使用动态长度）
        alpha_bytes = alpha_i.to_bytes((alpha_i.bit_length() + 7) // 8, 'big')
        data = alpha_bytes + PID_ij_t + self.group.serialize(kappa_ij)
        return self._hash_to_zq(data)
    '''
    def H7(self, C_ij, PID_ij_t: bytes, pk_ij, T_ij: bytes, Y_ij, tau_ij: bytes, Com_ij_t: bytes) -> bytes:
        data = (self.group.serialize(C_ij) + PID_ij_t + self.group.serialize(pk_ij) +
                T_ij + self.group.serialize(Y_ij) + tau_ij + Com_ij_t)
        return self._hkdf(data, self.OUT_BYTES, b"H7")
    '''
    def H7(self, C_ij, PID_ij_t: bytes, pk_ij, T_ij: bytes, Y_ij, tau_ij: bytes, Com_ij_t: bytes) -> bytes:
        X_ij, R_ij = pk_ij   # 解包公钥对
        data = (self._serialize_ciphertext(C_ij) + PID_ij_t +
                self.group.serialize(X_ij) + self.group.serialize(R_ij) +
                T_ij + self.group.serialize(Y_ij) + tau_ij + Com_ij_t)
        return self._hkdf(data, self.OUT_BYTES, b"H7")

    def H(self, PID_ij_t: bytes, T_ij: bytes, C_ij, omega_ij, Com_ij_t: bytes) -> bytes:
        data = PID_ij_t + T_ij + self._serialize_ciphertext(C_ij) + self.group.serialize(omega_ij) + Com_ij_t
        return self._hkdf(data, self.OUT_BYTES, b"H")

    def hash_to_zq(self, data: bytes) -> ZR:
        """将字节数据映射到 ZR 元素"""
        return self._hash_to_zq(data)

    # 用户注册相关（与原始 TRS_FL 相同）
    def user_register(self):
        sk_ij_puf = ''.join(secrets.choice('01') for _ in range(self.OUT_BYTES))
        sk_ij_puf_bytes = sk_ij_puf.encode()
        Com_ij = self.H1(sk_ij_puf_bytes, self.s)
        x_ij = self.group.random(ZR)
        X_ij = self.P ** x_ij
        return sk_ij_puf, Com_ij, x_ij, X_ij

    def pid_gen(self, Com_ij: bytes, X_ij) -> bytes:
        return self.H2(Com_ij, X_ij, self.P_pub)

    def ppk_gen(self, PID_ij: bytes, X_ij):
        r_ij = self.group.random(ZR)
        R_ij = self.P ** r_ij
        d_ij = self.x * self.H3(PID_ij, self.P_pub, X_ij, R_ij) + r_ij
        return d_ij, R_ij, r_ij

    def uk_gen(self, x_ij, d_ij, X_ij, R_ij):
        return (x_ij, d_ij), (X_ij, R_ij)

    def rpid_gen(self, sk_ij_puf: str, Com_ij: bytes, t: int, PID_ij: bytes):
        nonce_ij_t = ''.join(secrets.choice('01') for _ in range(self.OUT_BYTES))
        Com_ij_t = self.H4(nonce_ij_t, Com_ij)
        PID_ij_t = self.H5(PID_ij, t, Com_ij_t)
        return nonce_ij_t, Com_ij_t, PID_ij_t

# ==================== AS 类 ====================
class AS:
    def __init__(self, ta):
        self.ta = ta
        self.group = ta.group
        self.P = ta.P
        self.sk_AS = None
        self.pk_AS = None
        # 存储每个 ES 组的种子（秘密）
        self.group_seeds = {}  # key: es_index, value: seed (int)

    def keygen(self):
        self.sk_AS = self.group.random(ZR)
        self.pk_AS = self.P ** self.sk_AS
        return self.sk_AS, self.pk_AS

    def group_users_by_es(self, user_list, min_per_group):
        """
        将用户列表分成若干组（每组对应一个 ES），组数 = floor(n / min_per_group)，
        各组人数尽量均衡。
        返回嵌套列表：result[es_idx] = 该 ES 管理的用户列表。
        """
        n = len(user_list)
        if n < min_per_group:
            raise ValueError("总人数不能少于每组最少人数")
        m = n // min_per_group          # ES 数量
        base = n // m
        remainder = n % m
        sizes = [base + 1] * remainder + [base] * (m - remainder)

        result = []
        start = 0
        for size in sizes:
            result.append(user_list[start:start+size])
            start += size
        return result

    def setup_group_secrets(self, es_groups, l):
        """
        为每个 ES 组生成独立的秘密（种子），并通过 Asmuth-Bloom 共享给组内用户。
        返回一个字典：es_index -> (m0, moduli, shares_list)
        """
        group_params = {}
        for idx, group in enumerate(es_groups):
            n_i = len(group)
            # 生成一个随机秘密（种子），位数 500 位
            secret = random.getrandbits(100)
            m0, moduli, shares = share_secret(secret, n_i, l)
            # 存储种子供后续解签密使用
            self.group_seeds[idx] = secret
            # 将每个用户的份额和模数保存到对应的 user 对象中
            for j, user in enumerate(group):
                user.share = shares[j]
                user.modulus = moduli[j]
                user.m0 = m0
                user.l = l
            group_params[idx] = (m0, moduli, shares)
        return group_params
    def unsigncrypt_aggregate(self, sk_AS, members, open_list, open_ES):
        """
        逐坐标聚合 64 维密文并验证，去除掩码后恢复等权平均向量。
        :param sk_AS: AS 私钥 (ZR)
        :param members: 列表，每个元素包含用户假名等，例如 [{'PID': PID_ij}, ...]
        :param open_list: 列表，每个元素包含用户签密时的公开数据，如：
            {'Y': Y_ij, 'R': R_ij, 'X': X_ij, 'tau': tau_ij, 'C': C_ij,
            'PID': PID_ij_t, 'pk': pk_ij, 'T': T_ij, 'Com': Com_ij_t}
        :param open_ES: 列表，每个元素包含 ES 聚合结果，如 {'C_i': C_i, 'sigma_i': sigma_i}
        :return: (is_valid, W_g_t, verify_time)，W_g_t 为 64 维 ZR 向量
        """
        ta = self.ta
        group = ta.group
        P = ta.P
        P_pub = ta.P_pub

        if not open_list:
            raise ValueError("至少需要一个用户的签密数据")
        if len(members) != len(open_list):
            raise ValueError("成员信息与签密数据数量不一致")

        # 初始化聚合变量
        sigma_total = group.init(ZR, 0)
        C_total = [group.init(ZR, 0) for _ in range(64)]
        k_total = group.init(ZR, 0)  # 验签用的标量密钥和
        mask_total = [group.init(ZR, 0) for _ in range(64)]
        xi_Y = P ** 0          # 单位元
        xi_R = P ** 0
        Ppub_hxi = P ** 0

        # 逐坐标聚合 ES 密文，签名仍按标量聚合。
        for es_data in open_ES:
            if len(es_data['C_i']) != 64:
                raise ValueError("ES 聚合密文必须为 64 维向量")
            C_total = [c + value for c, value in zip(C_total, es_data['C_i'])]
            sigma_total += es_data['sigma_i']

        verify_times=[]
        # 遍历每个用户
        for j in range(len(open_list)):
            Y_ij = open_list[j]['Y']
            R_ij = open_list[j]['R']
            X_ij = open_list[j]['X']
            tau_ij = open_list[j]['tau']
            C_ij = open_list[j]['C']
            PID_ij_t = open_list[j]['PID']
            PID_ij = members[j]['PID']
            pk_ij = open_list[j]['pk']   # (X_ij, R_ij)
            T_ij = open_list[j]['T']
            Com_ij_t = open_list[j]['Com']
            alpha_i = open_list[j]['alpha_i']   # 每个用户的种子

            start_time = time.perf_counter()
            # 重新计算用户相关值
            kappa_ij_new = Y_ij ** sk_AS
            k_ij_new = ta.H6(alpha_i, PID_ij_t, kappa_ij_new)
            mask = ta.mask_vector(k_ij_new, PID_ij_t, Y_ij)
            xi_ij_new = ta.H7(C_ij, PID_ij_t, pk_ij, T_ij, Y_ij, tau_ij, Com_ij_t)
            # 注意 H7 返回 bytes，需转为 ZR
            xi_zr = ta.hash_to_zq(xi_ij_new)
            h_new = ta.H3(PID_ij, P_pub, X_ij, R_ij)

            k_total += k_ij_new
            mask_total = [total + value for total, value in zip(mask_total, mask)]
            xi_Y = xi_Y * (Y_ij ** xi_zr)
            Ppub_hxi = Ppub_hxi * (P_pub ** (h_new * xi_zr))
            xi_R = xi_R * (R_ij ** xi_zr)
            end_time = time.perf_counter()
            verify_times.append((end_time- start_time)*1000)
        sum_time =sum(verify_times)

        # 验证等式
        start_time=time.perf_counter()
        left = P ** sigma_total
        right = (P ** k_total) * xi_Y * Ppub_hxi * xi_R
        is_valid = (left == right)
        # 在 ZR 上逐坐标去掩码，再按实际参与人数求等权平均。
        inverse_n = group.init(ZR, pow(len(open_list), -1, int(ta.q)))
        W_g_t = [(c - mask_sum) * inverse_n
                 for c, mask_sum in zip(C_total, mask_total)]
        end_time = time.perf_counter()
        verify_time = (end_time - start_time) * 1000 +sum_time
        return is_valid, W_g_t, verify_time
# ==================== ES 类 ====================
class ES:
    def __init__(self, ta, es_id=None):
        self.ta = ta
        self.id = es_id
        self.group = ta.group
        self.P = ta.P
        self.sk_ES = None
        self.pk_ES = None

    def keygen(self):
        self.sk_ES = self.group.random(ZR)
        self.pk_ES = self.P ** self.sk_ES
        return self.sk_ES, self.pk_ES

    def edge_verify_aggregate(self, user_data_list):
        """
        验证来自用户的签密消息，逐坐标聚合密文并累加标量签名。
        user_data_list: 列表，每个元素包含 tau_ij, PID_ij_t, T_ij, C_ij, Com_ij_t, Y_ij, sigma_ij
        """
        C_i = [self.group.init(ZR, 0) for _ in range(64)]
        sigma_i = self.group.init(ZR, 0)
        all_valid = True
        for data in user_data_list:
            omega_ij_new = data['Y_ij'] ** self.sk_ES
            expected_tau = self.ta.H(
                data['PID_ij_t'], data['T_ij'], data['C_ij'],
                omega_ij_new, data['Com_ij_t']
            )
            if data['tau_ij'] != expected_tau:
                all_valid = False
            C_i = [total + value for total, value in zip(C_i, data['C_ij'])]
            sigma_i += data['sigma_ij']
        return all_valid, C_i, sigma_i

# ==================== User 类 ===================
class User:
    def __init__(self, ta, user_id=None):
        self.ta = ta
        self.id = user_id
        self.group = ta.group
        self.P = ta.P

        # 注册阶段
        self.sk_ij_puf, self.Com_ij, self.x_ij, self.X_ij = ta.user_register()
        self.PID_ij = ta.pid_gen(self.Com_ij, self.X_ij)
        self.d_ij, self.R_ij, self.r_ij = ta.ppk_gen(self.PID_ij, self.X_ij)
        self.sk_ij, self.pk_ij = ta.uk_gen(self.x_ij, self.d_ij, self.X_ij, self.R_ij)

        # 动态身份（稍后设置）
        self.nonce_ij_t = None
        self.Com_ij_t = None
        self.PID_ij_t = None

        # 轮密钥相关
        self.y_ij = None
        self.Y_ij = None
        self.k_ij = None
        self.W_ij = None

        # 秘密共享相关（由 AS 填充）
        self.share = None      # 自己的份额
        self.modulus = None    # 自己的模数 m_i
        self.m0 = None         # 公开模数
        self.l = None          # 门限
        self.group_seed = None # 重构后的秘密（alpha_i）

    def generate_dynamic_id(self, t: int):
        self.nonce_ij_t, self.Com_ij_t, self.PID_ij_t = self.ta.rpid_gen(
            self.sk_ij_puf, self.Com_ij, t, self.PID_ij
        )
        return self.PID_ij_t

    def reconstruct_seed(self, other_shares_with_moduli):
        """
        使用自己的份额和从其他用户收集的份额（至少 l-1 个）重构组秘密。
        other_shares_with_moduli: 列表 [(share, modulus), ...] 其他用户的份额及模数
        """                     
        if self.l == 1:
            pairs = [(self.share, self.modulus)]
            self.group_seed = reconstruct_secret(pairs, self.m0)
            return self.group_seed
        else:
            # 收集至少 l 个碎片（包括自己的）
            pairs = [(self.share, self.modulus)] + other_shares_with_moduli
            if len(pairs) < self.l:
                raise ValueError(f"碎片不足 {self.l} 个，无法重构")
            # 只取前 self.l 个进行重构（实际应取任意 self.l 个，这里简单取前 self.l 个）
            pairs = pairs[:self.l]
            seed = reconstruct_secret(pairs, self.m0)
            self.group_seed = seed
            return seed

    def generate_round_key(self, pk_AS):
        if self.group_seed is None:
            raise ValueError("请先重构组秘密")
        self.y_ij = self.group.random(ZR)
        self.Y_ij = self.X_ij ** self.y_ij
        kappa_ij = pk_AS** (self.x_ij * self.y_ij)   
        self.k_ij = self.ta.H6(self.group_seed, self.PID_ij_t, kappa_ij)
        return self.Y_ij, self.k_ij

    def train(self):
        self.W_ij = [self.group.random(ZR) for _ in range(64)]
        return self.W_ij

    def signcrypt(self, pk_ES):
        if self.W_ij is None or self.k_ij is None:
            raise ValueError("请先调用 train() 和 generate_round_key()")
        if len(self.W_ij) != 64:
            raise ValueError("梯度必须为 64 维向量")
        omega_ij = pk_ES**(self.x_ij * self.y_ij)
        mask = self.ta.mask_vector(self.k_ij, self.PID_ij_t, self.Y_ij)
        C_ij = [w + r for w, r in zip(self.W_ij, mask)]
        T_ij = int(time.time()).to_bytes(32, 'big')
        tau_ij = self.ta.H(self.PID_ij_t, T_ij, C_ij, omega_ij, self.Com_ij_t)
        xi_bytes = self.ta.H7(C_ij, self.PID_ij_t, self.pk_ij, T_ij, self.Y_ij, tau_ij, self.Com_ij_t)
        xi_ij = self.ta.hash_to_zq(xi_bytes)
        sigma_ij = self.k_ij + (self.y_ij * self.x_ij + self.d_ij) * xi_ij
        return C_ij, sigma_ij, T_ij, tau_ij
# ==================== 主函数 ====================
def main():
    print("=== 系统初始化 ===")
    ta = TA()
    as_server = AS(ta)
    as_server.keygen()

    # 参数
    total_users = int(input("请输入总用户数 n : "))
    min_per_group = int(input("请输入门限l:"))             # l
    t = 1                                            # 动态身份时间参数

    print(f"总用户数: {total_users}, 每组最少人数: {min_per_group}")

    # 1. 生成所有用户对象
    users = [User(ta, f"user_{i:03d}") for i in range(total_users)]

    # 2. AS 分组（确定 ES 数量及每个 ES 的用户列表）
    es_groups = as_server.group_users_by_es(users, min_per_group)
    num_es = len(es_groups)
    print(f"自动分成 {num_es} 个 ES 组")
    for idx, group in enumerate(es_groups):
        print(f"  ES_{idx+1}: {len(group)} 个用户")

    # 3. 创建 ES 实例并生成密钥对
    es_list = [ES(ta, f"ES_{i+1}") for i in range(num_es)]
    for es in es_list:
        es.keygen()

    # 4. AS 为每个 ES 组生成独立的秘密种子，并分发份额给用户
    print("\n=== 为每个 ES 组生成秘密种子及份额 ===")
    as_server.setup_group_secrets(es_groups, min_per_group)

    # 5. 模拟用户从同组其他用户收集碎片并重构种子
    #    为了简化，每个用户从所在组的其他用户中随机选取 l-1 个用户的份额进行重构
    for idx, group in enumerate(es_groups):
        # print(f"\nES 组 {idx+1} 内用户重构种子:")
        # 收集该组所有用户的 (user, share, modulus)
        members = [(user, user.share, user.modulus) for user in group]
        for user in group:
            # 选择其他 l-1 个用户（如果组内人数大于门限，否则用全部）
            others = [m for m in members if m[0] != user]
            needed = user.l - 1
            if len(others) < needed:
                # print(f"  用户 {user.id} 所在组人数不足 {user.l}，无法重构")
                continue
            chosen = random.sample(others, needed)
            other_pairs = [(share, mod) for (_, share, mod) in chosen]
            seed = user.reconstruct_seed(other_pairs)
            # print(f"  用户 {user.id} 重构种子 = {seed} (应与组种子一致)")
        # 验证：组种子是否与 AS 存储的一致
        expected_seed = as_server.group_seeds[idx]
        # print(f"  AS 存储的该组种子 = {expected_seed}")

    # 6. 每个用户生成动态假名、轮密钥、训练、签密，并发送给对应的 ES
    #    首先建立用户到 ES 的映射
    members = []           # 每个用户的基本信息（如 PID_ij）
    open_list = []         # 每个用户的签密公开数据
    alpha_i_list = []      # 每个用户重构的种子

    user_to_es = {}
    for es_idx, group in enumerate(es_groups):
        for user in group:
            user_to_es[user] = es_list[es_idx]

    # 存储每个 ES 接收到的签密数据
    es_received = {es: [] for es in es_list}

    print("\n=== 用户执行签密并发送给 ES ===")
    stime=[]
    for user in users:
        es = user_to_es[user]
        user.generate_dynamic_id(t)
        # 确保用户已重构种子
        if user.group_seed is None:
            print(f"用户 {user.id} 未重构种子，跳过")
            continue
        user.generate_round_key(as_server.pk_AS)
        user.train()
        start_time = time.time()
        C_ij, sigma_ij, T_ij, tau_ij = user.signcrypt(es.pk_ES)
        end_time = time.time()
        stime.append(end_time - start_time)
        # 记录 members
        members.append({'PID': user.PID_ij})
        # 记录 open_list
        open_list.append({
            'Y': user.Y_ij,
            'R': user.R_ij,
            'X': user.X_ij,
            'tau': tau_ij,
            'C': C_ij,
            'PID': user.PID_ij_t,
            'pk': user.pk_ij,
            'T': T_ij,
            'Com': user.Com_ij_t,
            'alpha_i': user.group_seed
        })
        alpha_i_list.append(user.group_seed)
        es_received[es].append({
            'tau_ij': tau_ij,
            'PID_ij_t': user.PID_ij_t,
            'T_ij': T_ij,
            'C_ij': C_ij,
            'Com_ij_t': user.Com_ij_t,
            'Y_ij': user.Y_ij,
            'sigma_ij': sigma_ij,
            'user': user
        })
    print(f"平均签密时间: {1000*(sum(stime)/len(stime)):.4f} ms")

    # 7. 每个 ES 验证并聚合
    print("\n=== ES 验证并聚合签密消息 ===")
    es_aggregates = []
    for es, data_list in es_received.items():
        valid, C_i, sigma_i = es.edge_verify_aggregate(data_list)
        print(f"{es.id}: 验证结果 = {valid} ")  # 聚合 C_i = {C_i}, sigma_i = {sigma_i}
        es_aggregates.append({
            'es_id': es.id,
            'C_i': C_i,
            'sigma_i': sigma_i,
            'data_list': data_list
        })
    open_ES = []
    for agg in es_aggregates:
        open_ES.append({
            'C_i': agg['C_i'],
            'sigma_i': agg['sigma_i']
        })

    print("\n=== AS 聚合验证并解签密 ===")
    Vtime=[]
    for i in range(10):
        
        is_valid, W_g_t,verify_time = as_server.unsigncrypt_aggregate(
            as_server.sk_AS, members, open_list, open_ES
        )
        Vtime.append(verify_time)   
    mean_time = sum(Vtime) / len(Vtime)
    print(f"聚合验证时间: {mean_time:.4f} ms")
    print(f"聚合签名验证结果: {is_valid}")
    print(f"恢复的全局训练结果 W_g_t = {W_g_t}")
def measure_aggregate_no_group(n, l=None, repetitions=3):
    """
    不分组版本（传统）：所有用户属于同一个 ES。
    n: 用户数
    l: 忽略（仅为了接口统一）
    repetitions: 重复次数
    """
    ta = TA()
    as_server = AS(ta)
    as_server.keygen()
    t = 1

    total_times = []
    for _ in range(repetitions):
        users = [User(ta, f"user_{i:03d}") for i in range(n)]
        # 强制只有一个组
        es_groups = [users]   # 所有用户给同一个 ES
        num_es = 1
        es_list = [ES(ta, "ES_single")]
        for es in es_list:
            es.keygen()
        as_server.setup_group_secrets(es_groups, l=1)  # 秘密共享门限无关

        user_to_es = {user: es_list[0] for user in users}

        # 重构种子（简化，每个用户用自己的份额）
        for user in users:
            user.reconstruct_seed([])

        members, open_list, es_received = build_es_data(users, user_to_es, as_server, t)

        start = time.perf_counter()
        es_aggregates = []
        for es, data_list in es_received.items():
            valid, C_i, sigma_i = es.edge_verify_aggregate(data_list)
            es_aggregates.append({'C_i': C_i, 'sigma_i': sigma_i})
        open_ES = [{'C_i': agg['C_i'], 'sigma_i': agg['sigma_i']} for agg in es_aggregates]
        is_valid, _ , _= as_server.unsigncrypt_aggregate(
            as_server.sk_AS, members, open_list, open_ES
        )
        total_times.append((time.perf_counter() - start) * 1000)

    return sum(total_times) / repetitions

def measure_aggregate_with_group(n, l, repetitions=3):
    """
    分组版本（TRS-FL 原始逻辑）
    """
    ta = TA()
    as_server = AS(ta)
    as_server.keygen()
    t = 1
    total_times = []
    for _ in range(repetitions):
        users = [User(ta, f"user_{i:03d}") for i in range(n)]
        es_groups = as_server.group_users_by_es(users, l)
        num_es = len(es_groups)
        es_list = [ES(ta, f"ES_{i+1}") for i in range(num_es)]
        for es in es_list:
            es.keygen()
        as_server.setup_group_secrets(es_groups, l)

        user_to_es = {}
        for es_idx, group in enumerate(es_groups):
            for user in group:
                user_to_es[user] = es_list[es_idx]

        for user in users:
            my_group = next(g for g in es_groups if user in g)
            others = [(u.share, u.modulus) for u in my_group if u != user]
            user.reconstruct_seed(others[:max(0, user.l-1)])

        members, open_list, es_received = build_es_data(users, user_to_es, as_server, t)

        start = time.perf_counter()
        es_aggregates = []
        for es, data_list in es_received.items():
            valid, C_i, sigma_i = es.edge_verify_aggregate(data_list)
            es_aggregates.append({'C_i': C_i, 'sigma_i': sigma_i})
        open_ES = [{'C_i': agg['C_i'], 'sigma_i': agg['sigma_i']} for agg in es_aggregates]
        is_valid, _, _= as_server.unsigncrypt_aggregate(
            as_server.sk_AS, members, open_list, open_ES
        )
        total_times.append((time.perf_counter() - start) * 1000)

    return sum(total_times) / repetitions

def build_es_data(users, user_to_es, as_server, t):
    """辅助函数：生成每个用户的签密数据，按 ES 分组"""
    members = []
    open_list = []
    es_received = {es: [] for es in set(user_to_es.values())}
    for user in users:
        es = user_to_es[user]
        user.generate_dynamic_id(t)
        user.generate_round_key(as_server.pk_AS)
        user.train()
        C_ij, sigma_ij, T_ij, tau_ij = user.signcrypt(es.pk_ES)
        members.append({'PID': user.PID_ij})
        open_list.append({
            'Y': user.Y_ij, 'R': user.R_ij, 'X': user.X_ij, 'tau': tau_ij,
            'C': C_ij, 'PID': user.PID_ij_t, 'pk': user.pk_ij,
            'T': T_ij, 'Com': user.Com_ij_t, 'alpha_i': user.group_seed
        })
        es_received[es].append({
            'tau_ij': tau_ij, 'PID_ij_t': user.PID_ij_t, 'T_ij': T_ij,
            'C_ij': C_ij, 'Com_ij_t': user.Com_ij_t, 'Y_ij': user.Y_ij,
            'sigma_ij': sigma_ij, 'user': user
        })
    return members, open_list, es_received
#测试代码：测量签名生成和验证时间
def measure(repetitions=10):
    """
    测量 TRS-FL 方案在 n=1, l=1 条件下的签名时间和验证时间（毫秒）
    签名时间：用户 signcrypt 耗时
    验证时间：ES 聚合验证 + AS 聚合验证的总耗时（单个用户）
    :param repetitions: 重复测量次数，取平均值
    :return: (avg_sign_time_ms, avg_verify_time_ms)
    """
    ta = TA()
    as_server = AS(ta)
    as_server.keygen()

    total_users = 1
    l = 1                     # 门限值
    t = 1

    users = [User(ta, f"user_{i:03d}") for i in range(total_users)]

    # 分组（仅一组，包含这一个用户）
    es_groups = as_server.group_users_by_es(users, l)
    num_es = len(es_groups)
    es_list = [ES(ta, f"ES_{i+1}") for i in range(num_es)]
    for es in es_list:
        es.keygen()

    # AS 为每个 ES 组生成秘密种子及份额
    as_server.setup_group_secrets(es_groups, l)

    # 建立用户到 ES 的映射
    user_to_es = {}
    for es_idx, group in enumerate(es_groups):
        for user in group:
            user_to_es[user] = es_list[es_idx]

    sign_times = []
    verify_times = []

    for _ in range(repetitions):
        # 每个用户重构种子（l=1 时只需要自己的份额）
        for user in users:
            user.reconstruct_seed([])   # 空列表表示不需要其他用户份额

        # 准备收集数据
        members = []
        open_list = []
        es_received = {es: [] for es in es_list}

        # 用户执行签密
        for user in users:
            es = user_to_es[user]
            user.generate_dynamic_id(t)
            user.generate_round_key(as_server.pk_AS)
            user.train()
            start_sign = time.perf_counter()
            C_ij, sigma_ij, T_ij, tau_ij = user.signcrypt(es.pk_ES)
            sign_time = (time.perf_counter() - start_sign) * 1000
            sign_times.append(sign_time)

            members.append({'PID': user.PID_ij})
            open_list.append({
                'Y': user.Y_ij,
                'R': user.R_ij,
                'X': user.X_ij,
                'tau': tau_ij,
                'C': C_ij,
                'PID': user.PID_ij_t,
                'pk': user.pk_ij,
                'T': T_ij,
                'Com': user.Com_ij_t,
                'alpha_i': user.group_seed
            })
            es_received[es].append({
                'tau_ij': tau_ij,
                'PID_ij_t': user.PID_ij_t,
                'T_ij': T_ij,
                'C_ij': C_ij,
                'Com_ij_t': user.Com_ij_t,
                'Y_ij': user.Y_ij,
                'sigma_ij': sigma_ij,
                'user': user
            })

        # ES 验证聚合
        es_aggregates = []
        for es, data_list in es_received.items():
            valid, C_i, sigma_i = es.edge_verify_aggregate(data_list)
            es_aggregates.append({
                'es_id': es.id,
                'C_i': C_i,
                'sigma_i': sigma_i,
                'data_list': data_list
            })
        open_ES = [{'C_i': agg['C_i'], 'sigma_i': agg['sigma_i']} for agg in es_aggregates]

        # AS 聚合验证
        is_valid, W_g_t, verify_time= as_server.unsigncrypt_aggregate(
            as_server.sk_AS, members, open_list, open_ES
        )
        verify_times.append(verify_time)
        assert is_valid, "聚合验证失败"

    avg_sign = sum(sign_times) / repetitions
    avg_verify = sum(verify_times) / repetitions
    return avg_sign, avg_verify

def measure_aggregate(n, l=1, repetitions=3):
    """
    测量 TRS-FL 方案在 n 个用户、门限 l 下的聚合验证总时间（毫秒）
    包括：ES 边缘验证 + AS 聚合验证的完整流程
    :param n: 用户总数
    :param l: 门限值 (建议设为 1，简化重构)
    :param repetitions: 重复次数，取平均
    :return: 平均聚合验证时间 (ms)
    """
    ta = TA()
    as_server = AS(ta)
    as_server.keygen()
    t = 1

    total_times = []
    for _ in range(repetitions):
        # 创建用户
        users = [User(ta, f"user_{i:03d}") for i in range(n)]

        # 分组（确保每个组人数 >= l）
        if n < l:
            raise ValueError("n must be >= l")
        es_groups = as_server.group_users_by_es(users, l)
        num_es = len(es_groups)
        es_list = [ES(ta, f"ES_{i+1}") for i in range(num_es)]
        for es in es_list:
            es.keygen()

        # AS 生成秘密种子和份额
        as_server.setup_group_secrets(es_groups, l)

        # 用户到 ES 的映射
        user_to_es = {}
        for es_idx, group in enumerate(es_groups):
            for user in group:
                user_to_es[user] = es_list[es_idx]

        # 每个用户重构种子（门限 l=1 时不需要其他用户份额）
        for user in users:
            my_group = next(g for g in es_groups if user in g)
            others = [(u.share, u.modulus) for u in my_group if u != user]
            needed = user.l - 1
            if len(others) >= needed:
                user.reconstruct_seed(others[:needed])
            else:
                user.reconstruct_seed(others)   # 不够时就用已有的

        # 准备数据容器
        members = []
        open_list = []
        es_received = {es: [] for es in es_list}

        # 用户执行签密
        for user in users:
            es = user_to_es[user]
            user.generate_dynamic_id(t)
            user.generate_round_key(as_server.pk_AS)
            user.train()
            C_ij, sigma_ij, T_ij, tau_ij = user.signcrypt(es.pk_ES)

            members.append({'PID': user.PID_ij})
            open_list.append({
                'Y': user.Y_ij,
                'R': user.R_ij,
                'X': user.X_ij,
                'tau': tau_ij,
                'C': C_ij,
                'PID': user.PID_ij_t,
                'pk': user.pk_ij,
                'T': T_ij,
                'Com': user.Com_ij_t,
                'alpha_i': user.group_seed
            })
            es_received[es].append({
                'tau_ij': tau_ij,
                'PID_ij_t': user.PID_ij_t,
                'T_ij': T_ij,
                'C_ij': C_ij,
                'Com_ij_t': user.Com_ij_t,
                'Y_ij': user.Y_ij,
                'sigma_ij': sigma_ij,
                'user': user
            })

      
        # ES 验证聚合
        es_aggregates = []
        for es, data_list in es_received.items():
            valid, C_i, sigma_i = es.edge_verify_aggregate(data_list)
            if not valid:
                print(f"Warning: ES {es.id} verification failed")
            es_aggregates.append({'C_i': C_i, 'sigma_i': sigma_i})

        open_ES = [{'C_i': agg['C_i'], 'sigma_i': agg['sigma_i']} for agg in es_aggregates]

        # AS 聚合验证
        is_valid, W_g_t, verify_time = as_server.unsigncrypt_aggregate(
            as_server.sk_AS, members, open_list, open_ES
        )
        if not is_valid:
            print("Warning: AS aggregate verification failed")

        total_times.append(verify_time)   # 毫秒

    return sum(total_times) / repetitions

if __name__ == "__main__":
    main()
