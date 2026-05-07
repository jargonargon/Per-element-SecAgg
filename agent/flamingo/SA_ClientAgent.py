from agent.Agent import Agent
from agent.flamingo.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message

import dill
import time
import logging

import math
import libnum
import numpy as np
import pandas as pd
import random

# pycryptodomex library functions
from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS

# other user-level crypto functions
import hashlib
from util import param
from util import util
from util.crypto import ecchash
from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int
from pympler import asizeof
from pyroaring import BitMap
import zlib
import zstandard as zstd
import msgpack

# The PPFL_TemplateClientAgent class inherits from the base Agent class.
class SA_ClientAgent(Agent):
    
    def __str__(self):
        return "[client]"

    # Default param:
    # num of iterations = 4
    # key length = 32 bytes
    # neighbors ~ 2 * log(num per iter) 
    def __init__(self, id, name, type,
                 iterations=4,
                 key_length=32,  
                 num_clients=128,
                 neighborhood_size=1,
                 debug_mode=0,
                 random_state=None):

        # Base class init
        super().__init__(id, name, type, random_state)

        # Set logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if debug_mode:
            logging.basicConfig()


        """Read keys."""
        # Read system-wide pk
        self.system_pk = util.read_pk(f"pki_files/system_pk.pem")
        
        # sk is used to establish pairwise secret with neighbors' public keys
        self.key = util.read_key(f"pki_files/client{self.id}.pem")
        self.secret_key = self.key.d
        
        
        """Set parameters."""
        self.num_clients = num_clients
        self.neighborhood_size = neighborhood_size
        self.vector_len = param.vector_len
        self.extraMask_len = param.extraMask_len
        self.vector_dtype = param.vector_type
        self.prime = ecchash.n
        self.key_length = key_length
        self.neighbors_list = set() # neighbors
        self.cipher_stored = None   # Store cipher from server across steps
        
        # ICDCS: いくつのブロックに分割するか．


        """Select committee."""
        self.user_committee = param.choose_committee(param.root_seed, 
                                                     param.committee_size, 
                                                     self.num_clients)
        self.committee_shared_sk = None
        self.committee_member_idx = None

        """復号者のサブグループを導出"""
        self.user_subgroups_committee = param.choose_subgroups(param.root_seed,
                                                               self.user_committee,
                                                               param.numBlock)

        # If it is in the committee:
        # read pubkeys of every other client and precompute pairwise keys
        self.symmetric_keys = {}
        self.committee_chacha_keys = {}
        if self.id in self.user_committee:
            for i in range(num_clients):
                pk = util.read_pk(f"pki_files/client{i}.pem")

                # ECC-DH による共有点
                shared_point = pk * self.secret_key

                # AES 用 128bit 鍵（X座標の下位128bit）
                self.symmetric_keys[i] = (int(shared_point.x) & ((1 << 128) - 1)).to_bytes(16, 'big')

                # ChaCha20 用 256bit 鍵（X+Y を SHA256 → 先頭32バイト）
                px = int(shared_point.x).to_bytes(self.key_length, 'big')
                py = int(shared_point.y).to_bytes(self.key_length, 'big')
                hash_object = SHA256.new(data=(px+py))
                self.committee_chacha_keys[i] = hash_object.digest()[0:self.key_length]

        # Accumulate this client's run time information by step.
        self.elapsed_time = {'REPORT': pd.Timedelta(0),
                             'CROSSCHECK': pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             'EXTRA_DEC': pd.Timedelta(0),
                             }
        
        # Accumulate this client's communication size by step.
        self.message_size = { 'REPORT' : 0,
                              'LABEL&CIPHERTEXT' : 0,
                              'RECONSTRUCTION' : 0,
                              'EXTRA_RECV': 0,
                              'EXTRA_SEND' : 0,
                            }

        # Iteration counter
        self.no_of_iterations = iterations
        self.current_iteration = 1
        self.current_base = 0

        # State flag
        self.setup_complete = False


    # Simulation lifecycle messages.
    def kernelStarting(self, startTime):

        # Initialize custom state properties into which we will later accumulate results.
        # To avoid redundancy, we allow only the first client to handle initialization.
        if self.id == 0:
            self.kernel.custom_state['clt_report'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_crosscheck'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_reconstruction'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_extraDec'] = pd.Timedelta(0)

            ## Send communication size
            self.kernel.custom_state['clt_comm_report'] = 0
            self.kernel.custom_state['dec_comm_reconstruction'] = 0
            self.kernel.custom_state['dec_comm_extraRecv'] = 0
            
            ## Receive communication size
            self.kernel.custom_state['dec_comm_label&cipher'] = 0
            self.kernel.custom_state['dec_comm_extraSend'] = 0


        # Find the PPFL service agent, so messages can be directed there.
        self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)

        self.setComputationDelay(0)

        # Request a wake-up call as in the base Agent.  Noise is kept small because
        # the overall protocol duration is so short right now.  (up to one microsecond)
        super().kernelStarting(startTime +
                               pd.Timedelta(self.random_state.randint(low=0, high=1000), unit='ns'))

    def kernelStopping(self):

        # Accumulate into the Kernel's "custom state" this client's elapsed times per category.
        # Note that times which should be reported in the mean per iteration are already so computed.
        # These will be output to the config (experiment) file at the end of the simulation.

        self.kernel.custom_state['clt_report'] += (
            self.elapsed_time['REPORT'] / self.no_of_iterations)
        self.kernel.custom_state['clt_crosscheck'] += (
            self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['clt_reconstruction'] += (
            self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)
        self.kernel.custom_state['clt_extraDec'] += (
            self.elapsed_time['EXTRA_DEC'] / self.no_of_iterations)
        
        self.kernel.custom_state['clt_comm_report'] += self.message_size['REPORT'] / self.no_of_iterations
        self.kernel.custom_state['dec_comm_label&cipher'] += self.message_size['LABEL&CIPHERTEXT'] / self.no_of_iterations
        self.kernel.custom_state['dec_comm_reconstruction'] += self.message_size['RECONSTRUCTION'] / self.no_of_iterations
        self.kernel.custom_state['dec_comm_extraRecv'] += self.message_size['EXTRA_RECV'] / self.no_of_iterations
        self.kernel.custom_state['dec_comm_extraSend'] += self.message_size['EXTRA_SEND'] / self.no_of_iterations

        super().kernelStopping()

    # Simulation participation messages.
    def wakeup(self, currentTime):
        super().wakeup(currentTime)        
        self.sendVectors(currentTime)


    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        # with signatures of other clients from the server
        if msg.body['msg'] == "COMMITTEE_SHARED_SK":
            self.committee_shared_sk = msg.body['sk_share']
            self.committee_member_idx = msg.body['committee_member_idx']

        elif msg.body['msg'] == "SIGN":
            if msg.body['iteration'] == self.current_iteration:
                self.recordBandwidth(msg, 'LABEL&CIPHERTEXT')
                dt_protocol_start = pd.Timestamp('now')
                self.cipher_stored = msg
                
                self.signSendLabels(currentTime, msg.body['labels'])
                self.recordTime(dt_protocol_start, 'CROSSCHECK')

        elif msg.body['msg'] == "DEC":
            if msg.body['iteration'] == self.current_iteration:
                dt_protocol_start = pd.Timestamp('now')

                if self.cipher_stored == None:
                    if __debug__: self.logger.info("did not recv sign")
                else:
                    if self.cipher_stored.body['iteration'] == self.current_iteration:
                        self.decryptSendShares(util.deserialize_dim1_elgamal(self.cipher_stored.body['dec_target_pairwise']), 
                                               util.deserialize_tuples_bytes(self.cipher_stored.body['dec_target_mi']), 
                                               self.cipher_stored.body['client_id_list'],
                                               self.cipher_stored.body['b_iut_cipher']
                                               )
                self.recordTime(dt_protocol_start, 'RECONSTRUCTION')

        # End of the protocol / start the next iteration
        # Receiving the output from the server
        elif msg.body['msg'] == "REQ" and self.current_iteration != 0:
            # End of the iteration
            # Reset temp variables for each iteration
            
            # Enter next iteration
            self.current_iteration += 1
            if self.current_iteration > self.no_of_iterations:
                return

            dt_protocol_start = pd.Timestamp('now')
            self.sendVectors(currentTime)
            self.recordTime(dt_protocol_start, "REPORT")

        elif msg.body['msg'] == "EXTRA":
            # print(np.array(util.deserialize_tuples_bytes(self.cipher_stored.body['dec_target_riut'])).shape)
            self.recordBandwidth(msg,  "EXTRA_RECV")
            dt_protocol_start = pd.Timestamp('now')
            if self.cipher_stored.body['iteration'] == self.current_iteration:
                self.decryptEXTRAShares(msg.body['drop_decryptors_recon1'],
                                        self.cipher_stored.body['client_id_list'],
                                        util.deserialize_tuples_bytes(msg.body['dec_target_riut'])
                                        )
                self.recordTime(dt_protocol_start, "EXTRA_DEC")
                self.cipher_stored = None
 

    ###################################
    # Round logics
    ###################################
    def sendVectors(self, currentTime):
        dt_protocol_start = pd.Timestamp('now')
        dt_wake_start = pd.Timestamp('now')

        # Find this client's neighbors: parse graph from PRG(PRF(iter, root_seed))
        self.neighbors_list = param.findNeighbors(param.root_seed, self.current_iteration, self.num_clients, self.id, self.neighborhood_size)
        if __debug__:
            self.logger.info("client indices in neighbors list starts from 0")
            self.logger.info(f"client {self.id} neighbors list: {self.neighbors_list}")
     
        # Download public keys of neighbors from PKI file
        # Client index starting from 0
        neighbor_pubkeys = {}
        for id in self.neighbors_list:
            neighbor_pubkeys[id] = util.read_pk(f"pki_files/client{id}.pem")

        # send symmetric encryption of shares of mi  
        """
        individual private random seed mi is derived by using Cryptodome.Random (get_random_bytes) and changing it into integer.
        Here, key_length is set to 32 (default).
        """
        mi_bytes = get_random_bytes(self.key_length) 
        mi_number = int.from_bytes(mi_bytes, 'big')
        
        mi_shares = secret_int_to_points(secret_int=mi_number, 
            point_threshold=int(param.fraction * len(self.user_committee)), 
            num_points=len(self.user_committee), prime=self.prime)

        committee_pubkeys = {}
        for id in self.user_committee:
            committee_pubkeys[id] = util.read_pk(f"pki_files/client{id}.pem")
            
        """
        Encrypt shares of mi with symmetric encryption.
        """
        committee_pairwise_secret_group = {}
        committee_pairwise_secret_bytes = {}
        key_with_committee_bytes_dict = {}
        # separately encrypt each share
        enc_mi_shares = []
        # id is the x-axis
        cnt = 0
        for id in self.user_committee:
            ## transform share of mi_sghares[cnt]=(index, share) into fixed-length bytes.
            per_share_bytes = (mi_shares[cnt][1]).to_bytes(self.key_length, 'big') 
            
            # can be pre-computed
            ## derive shared key with ECC Diffie-Hellman
            committee_pairwise_secret_group[id] = self.secret_key * committee_pubkeys[id]

            px = (int(committee_pairwise_secret_group[id].x)).to_bytes(self.key_length, 'big')
            py = (int(committee_pairwise_secret_group[id].y)).to_bytes(self.key_length, 'big')
            hash_object = SHA256.new(data=(px+py))
            committee_pairwise_secret_bytes[id] = hash_object.digest()[0:self.key_length] 

            ## cut 128bit from x-coordinate and use it for AES key
            key_with_committee_bytes = (int(committee_pairwise_secret_group[id].x) & ((1<<128)-1)).to_bytes(16, 'big')
            key_with_committee_bytes_dict[id] = key_with_committee_bytes

            per_share_encryptor = AES.new(key_with_committee_bytes, AES.MODE_GCM)
            # nouce should be sent with ciphertext
            nonce = per_share_encryptor.nonce
        
            tmp, _ = per_share_encryptor.encrypt_and_digest(per_share_bytes)
            enc_mi_shares.append((tmp, nonce))
            cnt += 1

        # Compute mask, compute masked vector
        # PRG individual mask
        prg_mi_holder = ChaCha20.new(key=mi_bytes, nonce=param.nonce)
        data = param.fixed_key * self.vector_len
        prg_mi = prg_mi_holder.encrypt(data)

        # compute pairwise masks r_ij
        neighbor_pairwise_secret_group = {}  # g^{a_i a_j} = r_ij in group
        neighbor_pairwise_secret_bytes = {}  
        
        for id in self.neighbors_list:
            neighbor_pairwise_secret_group[id] = self.secret_key * neighbor_pubkeys[id]
            # hash the g^{ai aj} to 256 bits (16 bytes)
            px = (int(neighbor_pairwise_secret_group[id].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_secret_group[id].y)).to_bytes(self.key_length, 'big')
            
            hash_object = SHA256.new(data=(px+py))
            neighbor_pairwise_secret_bytes[id] = hash_object.digest()[0:self.key_length] 
          

        neighbor_pairwise_mask_seed_group = {}
        neighbor_pairwise_mask_seed_bytes = {}

        """Mapping group elements to bytes.
            compute h_{i, j, t} to be PRF(r_ij, t)
            map h (a binary string) to a EC group element
            encrypt the group element
            map the group element to binary string (hash the x, y coordinate)
        """
        for id in self.neighbors_list:
            
            round_number_bytes = self.current_iteration.to_bytes(16, 'big')
            
            h_ijt = ChaCha20.new(key=neighbor_pairwise_secret_bytes[id], nonce=param.nonce).encrypt(round_number_bytes)
            h_ijt = str(int.from_bytes(h_ijt[0:4], 'big') & 0xFFFF)
         
            # map h_ijt to a group element
            dst = ecchash.test_dst("P256_XMD:SHA-256_SSWU_RO_")
            neighbor_pairwise_mask_seed_group[id] = ecchash.hash_str_to_curve(msg=h_ijt, count=2, 
                                                    modulus=self.prime, degree=ecchash.m, blen=ecchash.L, 
                                                    expander=ecchash.XMDExpander(dst, hashlib.sha256, ecchash.k)) 
            
            px = (int(neighbor_pairwise_mask_seed_group[id].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_mask_seed_group[id].y)).to_bytes(self.key_length, 'big')
            
            hash_object = SHA256.new(data=(px+py))
            neighbor_pairwise_mask_seed_bytes[id] = hash_object.digest()[0:self.key_length]
          
        prg_pairwise = {}
        for id in self.neighbors_list:
            prg_pairwise_holder = ChaCha20.new(key=neighbor_pairwise_mask_seed_bytes[id], nonce=param.nonce)
            data = param.fixed_key * self.vector_len
            prg_pairwise[id] = prg_pairwise_holder.encrypt(data)
        
        """Client inputs.
            For machine learning, replace it with model weights.
            For testing, set to unit vector.
        """
        vec = np.ones(self.vector_len, dtype=self.vector_dtype)

        # ----- Additional Masking Process -----
        rng = np.random.default_rng(seed=1)
        b_it = rng.integers(0, 2, size=self.extraMask_len, dtype=np.uint8)

        # ブロック分割（できるだけ均等）
        blocks = np.array_split(b_it, param.numBlock)  # numBlock=3想定

        # ブロック長と、extraMask領域内での開始位置（オフセット）
        block_lens = [len(b) for b in blocks]
        block_offsets = np.cumsum([0] + block_lens[:-1]).tolist()  # [0, len(block0), len(block0)+len(block1), ...]
        if self.id % 20 == 0:
            print(f"size of each block: {block_lens}")

        # ブロックハッシュ（b_itから直接）
        def hash_bit_block(bit_arr: np.ndarray) -> bytes:
            # packbits は 8bit 境界まで0埋めするので、長さも混ぜて曖昧性回避
            packed = np.packbits(bit_arr, bitorder="big")  # dtype=uint8
            L = len(bit_arr).to_bytes(4, "big")
            return SHA256.new(data=L + packed.tobytes()).digest()  # 32 bytes

        block_hashes = [hash_bit_block(b) for b in blocks]  # g=0..2

        # vec の extraMask は末尾にある前提： [ ... | extraMask_len ]
        tail_start = self.vector_len - self.extraMask_len

        # τ
        round_bytes = self.current_iteration.to_bytes(16, "big")

        committee_pairwise_mask_seed_bytes = {}
        prg_committee_pairwise = {}
        r_iut_dict = {}

        # サブグループ単位でループ（gが0,1,2）
        for g, subgroup in enumerate(self.user_subgroups_committee):
            h_iut = block_hashes[g]
            block_len = block_lens[g]
            off = block_offsets[g]

            # このブロックに加算する vec のスライス範囲（末尾extraMaskの中の一部）
            sl = slice(tail_start + off, tail_start + off + block_len)

            for id in subgroup:
                # r_iut = PRF(key, τ || h_iut) で、r_iutサイズは16 bytes維持
                prf_input_16 = SHA256.new(data=round_bytes + h_iut).digest()[:16]

                r_iut = ChaCha20.new(
                    key=committee_pairwise_secret_bytes[id],
                    nonce=param.nonce
                ).encrypt(prf_input_16)  # 16 bytes

                r_iut_dict[id] = r_iut

                # seed化
                committee_pairwise_mask_seed_bytes[id] = SHA256.new(data=r_iut).digest()[:self.key_length]

                # PRG：このサブグループ(g)のブロック長だけ生成（uint32で block_len 要素）
                prg = ChaCha20.new(key=committee_pairwise_mask_seed_bytes[id], nonce=param.nonce)

                data = param.fixed_key * block_len  # fixed_key=4bytes -> 出力 4*block_len bytes
                prg_bytes = prg.encrypt(data)
                prg_committee_pairwise[id] = prg_bytes

                vec_prg = np.frombuffer(prg_bytes, dtype=np.uint32)  # 要素数 = block_len
                # vec 側 dtype が uint32 前提。違うなら astype などで揃えてください。
                vec[sl] += vec_prg

        enc_b_iut = []  # (ciphertext, nonce) を格納

        for id in self.user_committee:
            # --- committee u(id) に割り当てられているブロック g を特定 ---
            g_assigned = None
            for g, subgroup in enumerate(self.user_subgroups_committee):
                if id in subgroup:
                    g_assigned = g
                    break
            if g_assigned is None:
                continue  # 想定外：committee なのにどの subgroup にもいない場合

            # --- 平文化（ビット列ブロックを bytes に） ---
            bit_arr = blocks[g_assigned].astype(np.uint8, copy=False)
            packed = np.packbits(bit_arr, bitorder="big")  # 8bit境界に0埋めされ得る
            L = len(bit_arr).to_bytes(4, "big")            # 元のビット長を先頭に付与（復号側の曖昧性回避）
            plaintext = L + packed.tobytes()

            # --- 共通鍵 k_iu で AES-GCM 暗号化 ---
            key_bytes = key_with_committee_bytes_dict[id]  # receiver_id = id
            encryptor = AES.new(key_bytes, AES.MODE_GCM)
            ciphertext, _ = encryptor.encrypt_and_digest(plaintext)

            # 要件： (暗号文, nonce) で append
            # ※暗号文に tag を連結して保持（復号時に末尾16bytesを tag として分離）
            enc_b_iut.append((ciphertext, encryptor.nonce))
            # enc_b_iut.append(("abc", "def"))

        # =========================
        # ----- Secret Sharing and Encryption of r_iut -----
        # =========================
        threshold = param.SSThreshold
        num_points = len(self.user_committee)

        enc_riut_shares_2d = []  # sender-major 二次元構造
        for sender_id in self.user_committee:
            row = []
            r_iut_int = int.from_bytes(r_iut_dict[sender_id], 'big') % self.prime
            shares_riut = secret_int_to_points(r_iut_int, threshold, num_points, self.prime)

            for idx, receiver_id in enumerate(self.user_committee):
                share_bytes = shares_riut[idx][1].to_bytes(self.key_length, 'big')
                key_bytes = key_with_committee_bytes_dict[receiver_id]

                encryptor = AES.new(key_bytes, AES.MODE_GCM)
                nonce_riut = encryptor.nonce
                ciphertext_riut, _ = encryptor.encrypt_and_digest(share_bytes)

                row.append((ciphertext_riut, nonce_riut))
            enc_riut_shares_2d.append(row)

        flat_enc_riut = [item for sublist in enc_riut_shares_2d for item in sublist]

        # =========================
        # 以降、既存処理
        # =========================

        # vectorize bytes: 32 bit integer, 4 bytes per component
        vec_prg_mi = np.frombuffer(prg_mi, dtype=self.vector_dtype)
        if len(vec_prg_mi) != self.vector_len:
            raise RuntimeError("vector length error")
        vec += vec_prg_mi

        vec_prg_pairwise = {}
        for id in self.neighbors_list:
            vec_prg_pairwise[id] = np.frombuffer(prg_pairwise[id], dtype=self.vector_dtype)

            if len(vec_prg_pairwise[id]) != self.vector_len:
                raise RuntimeError("vector length error")
            if self.id < id:
                vec = vec + vec_prg_pairwise[id]
            elif self.id > id:
                vec = vec - vec_prg_pairwise[id]
            else:
                raise RuntimeError("id itself appears in its neighbor list")

        # compute encryption of H(t)^{r_ij} (already a group element), only for < relation
        cipher_msg = {}
        for id in self.neighbors_list:
            cipher_msg[(self.id, id)] = self.elgamal_enc_group(self.system_pk, neighbor_pairwise_mask_seed_group[id])

        if __debug__:
            client_comp_delay = pd.Timestamp('now') - dt_protocol_start
            self.logger.info(f"client {self.id} computation delay for vector: {client_comp_delay}")
            self.logger.info(f"client {self.id} sends vector at {currentTime + client_comp_delay}")

        temp_Message = Message({
            "msg": "VECTOR",
            "iteration": self.current_iteration,
            "sender": self.id,
            "vector": vec,
            "enc_b_iut": enc_b_iut,
            "enc_mi_shares": util.serialize_tuples_bytes(enc_mi_shares),
            "enc_pairwise": util.serialize_dim1_elgamal(cipher_msg),
            "enc_riut_shares": util.serialize_tuples_bytes(flat_enc_riut),
        })

        
        # Send the vector to the server
        self.sendMessage(self.serviceAgentID,
                         temp_Message,
                         tag="comm_key_generation")
        self.recordTime(dt_wake_start, "REPORT")

        self.recordBandwidth(temp_Message, 'REPORT')

  
    def signSendLabels(self, currentTime, msg_to_sign):

        msg_to_sign = dill.dumps(msg_to_sign)
        hash_container = SHA256.new(msg_to_sign)
        signer = DSS.new(self.key, 'fips-186-3')
        signature = signer.sign(hash_container)
        client_signed_labels = (msg_to_sign, signature)

        self.sendMessage(self.serviceAgentID,
                         Message({"msg": "SIGN",
                                  "iteration": self.current_iteration,
                                  "sender": self.id,
                                  "signed_labels": client_signed_labels,
                                  "committee_member_idx": self.committee_member_idx,
                                  "signed_labels": client_signed_labels,
                                  }),
                        tag="comm_sign_client")

    def get_my_subgroup_index(self):
        for idx, subgroup in enumerate(self.user_subgroups_committee):
            if self.id in subgroup:
                return idx
        raise ValueError("self.id is not found in any subgroup")

    def decryptSendShares(self, dec_target_pairwise, dec_target_mi, client_id_list, 
                          b_ut_cipher):
        
        dt_protocol_start = pd.Timestamp('now')

        # b_ut_cipherの復号化
        restored_arrays = []

        cnt = 0
        for id in client_id_list:
            sym_key = self.symmetric_keys[id]
            ciphertext, nonce = b_ut_cipher[cnt]

            cipher = AES.new(sym_key, AES.MODE_GCM, nonce=nonce)
            plaintext_bytes = cipher.decrypt(ciphertext)

            # --- ここから復元 ---
            L = int.from_bytes(plaintext_bytes[:4], "big")
            packed = np.frombuffer(plaintext_bytes[4:], dtype=np.uint8)

            bits = np.unpackbits(packed, bitorder="big")[:L]
            bits_uint16 = bits.astype(np.uint16)

            restored_arrays.append(bits_uint16)
            cnt += 1
        numContributions = np.sum(restored_arrays, axis=0)

        # しきい値を満たすindex集合
        satisfying_indices = np.where(numContributions >= param.accessThreshold)[0]

        # 復号者 self.id が属するサブグループを特定
        my_group_id = self.get_my_subgroup_index()
        print(f"committee {self.id} belongs to {my_group_id}")

        round_bytes = self.current_iteration.to_bytes(16, 'big')

        # 暗号化側 sendVectors() と同じハッシュ関数（b_iutブロックから直接ハッシュ）
        def hash_bit_block(bit_arr: np.ndarray) -> bytes:
            packed = np.packbits(bit_arr.astype(np.uint8, copy=False), bitorder="big")
            L = len(bit_arr).to_bytes(4, "big")
            return SHA256.new(data=L + packed.tobytes()).digest()  # 32 bytes

        # committee u が全クライアント i との seed を再現するための保存先（必要なら）
        prg_seed_dict = {}   # key: client_id, value: prg_seed(bytes)
        r_iut_dict = {}      # key: client_id, value: r_iut(16bytes)

        emk_ut = np.zeros(len(numContributions), dtype=param.vector_type)

        # ここで restored_arrays と client_id_list が同じ順序で対応している前提
        # restored_arrays[idx] は「クライアント i が committee u に送った b_iut（復号済みビット列）」
        for idx, cid in enumerate(client_id_list):
            # 1) b_iut（復号済）からハッシュ h_iut を計算
            h_iut = hash_bit_block(restored_arrays[idx])  # 32 bytes

            # 2) round_bytes と h_iut から 16 bytes を作る（暗号化側と同型）
            prf_input_16 = SHA256.new(data=round_bytes + h_iut).digest()[:16]

            # 3) 秘密値 chacha_key（u-i の共有鍵）で r_iut を再現（16 bytes）
            chacha_key = self.committee_chacha_keys[cid]  # 32 bytes 想定
            r_iut = ChaCha20.new(key=chacha_key, nonce=param.nonce).encrypt(prf_input_16)  # 16 bytes

            r_iut_dict[cid] = r_iut

            # 4) r_iut から prg_seed を導出（key_length bytes）
            prg_seed = SHA256.new(data=r_iut).digest()[:self.key_length]
            prg_seed_dict[cid] = prg_seed

            # 5) PRG 出力を生成（ブロックぶん）
            prg = ChaCha20.new(key=prg_seed, nonce=param.nonce)
            data = param.fixed_key * len(numContributions)
            out_bytes = prg.encrypt(data)
            prg_output = np.frombuffer(out_bytes, dtype=param.vector_type)

            # 6) しきい値を満たすインデックス集合 D_i のみに加算
            emk_ut[satisfying_indices] += prg_output[satisfying_indices]

        
        if self.committee_shared_sk == None:
            if __debug__:
                self.logger.info(f"Decryptor {self.committee_member_idx} is asked to decrypt, but does not have sk share.")
            self.sendMessage(self.serviceAgentID,
                         Message({"msg": "NO_SK_SHARE",
                                  "iteration": self.current_iteration,
                                  "sender": self.id,
                                  "shared_result": None,
                                  "committee_member_idx": None,
                                  }),
                         tag="no_sk_share")
            return 
            
        # CHECK SIGNATURES

        """Compute decryption of pairwise secrets.
            dec_target is a matrix
            just need to mult sk with each of the entry
            needs elliptic curve ops
        """
        dec_shares_pairwise = [] 
        dec_target_list_pairwise = list(dec_target_pairwise.values())
       
        for i in range(len(dec_target_list_pairwise)):
            c0 = dec_target_list_pairwise[i][0]
            dec_shares_pairwise.append(self.committee_shared_sk[1] * c0)
       

        """Compute decryption for mi shares.
            dec_target_mi is a list of AES ciphertext (with nonce)
            decrypt each entry of dec_target_mi
        """
        dec_shares_mi = []
        cnt = 0
        for id in client_id_list:
            sym_key = self.symmetric_keys[id]
            dec_entry = dec_target_mi[cnt]
            nonce = dec_entry[1]
            cipher_holder = AES.new(sym_key, AES.MODE_GCM, nonce=nonce)
            plaintext = cipher_holder.decrypt(dec_entry[0])
            plaintext = int.from_bytes(plaintext, 'big')
            dec_shares_mi.append(plaintext)
            cnt += 1

        
        clt_comp_delay = pd.Timestamp('now') - dt_protocol_start

        if __debug__:
            self.logger.info(f"[Decryptor] run time for reconstruction step: {clt_comp_delay}")

        temp_Message = Message({"msg": "SHARED_RESULT",
                                  "iteration": self.current_iteration,
                                  "sender": self.id,
                                  "shared_result_pairwise": util.serialize_dim1_ecp(dec_shares_pairwise),
                                  "shared_result_mi": util.serialize_dim1_list(dec_shares_mi),
                                  "committee_member_idx": self.committee_member_idx,
                                  "emk_ut": emk_ut,
                                  "subgroup": my_group_id
                                  })
        
        self.sendMessage(self.serviceAgentID,
                         temp_Message,
                         tag="comm_secret_sharing")
        self.recordBandwidth(temp_Message, 'RECONSTRUCTION')        


    def elgamal_enc_group(self, system_pk, ptxt_point):
        # the order of secp256r1
        n = ecchash.n
        
        # ptxt is in ECC group
        enc_randomness_bytes = get_random_bytes(32)
        enc_randomness = (int.from_bytes(enc_randomness_bytes, 'big')) % n

        # base point in secp256r1
        base_point = ECC.EccPoint(ecchash.Gx, ecchash.Gy)

        c0 = enc_randomness * base_point
        c1 = ptxt_point + (system_pk * enc_randomness)
        return (c0, c1)
    
    def extract_all_user_lists(self, index_mask_array):
        """
        index_mask_array: 各要素が {'index': int, 'mask': int} のリスト
        mask の 1 ビットごとにだけループすることで O(寄与数) に削減
        """
        idx2users = {}
        for row in index_mask_array:
            idx, mask = row['index'], row['mask']
            users = []
            while mask:
                lsb = mask & -mask
                uid = lsb.bit_length() - 1
                users.append(uid)
                mask ^= lsb
            idx2users[idx] = users
        return idx2users
    
    def decryptEXTRAShares(self, droplist, client_id_list, dec_target_riut):
        if len(droplist) > param.droplistSize:
            raise ValueError("Too many drops!")

        # expected_len = len(client_id_list) * len(droplist)
        # if len(dec_target_riut) != expected_len:
        #     raise ValueError("Length mismatch in dec_target_riut")

        # シンプルに share 値のみを並べる
        share_list = []

        idx = 0
        for _ in client_id_list:
            for sender_id in droplist:
                key_bytes = self.symmetric_keys[sender_id]

                ciphertext, nonce = dec_target_riut[idx]
                idx += 1

                decryptor = AES.new(key_bytes, AES.MODE_GCM, nonce=nonce)
                share_bytes = decryptor.decrypt(ciphertext)
                share_value = int.from_bytes(share_bytes, 'big')

                share_list.append(share_value)
        if self.id % 5 == 0:
            print(f"How many times decrypts:  {idx}  ?= OnlineClients x DropDecryptors")
        # print(np.array(share_list).shape)

        # 送信（share_list の順序：client_iごとに sender_v の share を並べる）
        temp_Message = Message({"msg": "EXTRA_SHARED_RESULT",
                                "iteration": self.current_iteration,
                                "sender": self.id,
                                "shared_result_riut": util.serialize_dim1_list(share_list),
                                "committee_member_idx": self.committee_member_idx
                                })

        self.sendMessage(self.serviceAgentID,
                         temp_Message,
                         tag="comm_key_generation")
        self.recordBandwidth(temp_Message, 'EXTRA_SEND')


# ======================== UTIL ========================

    
    def recordTime(self, startTime, categoryName):
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime
    
    def agent_print(*args, **kwargs):
        """
        Custom print function that adds a [Server] header before printing.

        Args:
            *args: Any positional arguments that the built-in print function accepts.
            **kwargs: Any keyword arguments that the built-in print function accepts.
        """
        print(*args, **kwargs)
    def recordBandwidth(self, msgobj, categoryName):
      self.message_size[categoryName] += len(msgpack.packb(msgobj.to_payload(), use_bin_type=True))