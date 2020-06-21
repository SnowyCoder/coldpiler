// Here is listed EVERY riscv instruction, I know I won't use them all
#![allow(dead_code)]

use core::mem;


#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CondOp {
    Eq,
    Ne,
    Lt,
    Ge,
    Ltu,
    Geu,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ArithOp {
    Add, Sub, ShiftLeftLogic, ShiftRightLogic, ShiftRightArithmetic,
    SetLessThan, SetLessThanUnsigned, Xor, Or, And
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ArithImmOp {
    Add, ShiftLeftLogic, ShiftRightLogic, ShiftRightArithmetic,
    SetLessThan, SetLessThanUnsigned, Xor, Or, And
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum DataType {
    Byte,
    Half,
    Word,
    ByteUnsigned,
    HalfUnsigned,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Instr {
    CondBranch {
        diff: i16,
        rs1: u8,
        rs2: u8,
        op: CondOp,
    },
    Load {
        to: u8,
        from_reg: u8,
        from_imm: i16,
        dtype: DataType,
    },
    Store {
        from: u8,
        to_reg: u8,
        to_imm: i16,
        dtype: DataType,
    },
    ArithmeticReg {
        rd: u8,
        rs1: u8,
        rs2: u8,
        op: ArithOp,
    },
    ArithmeticImm {
        rd: u8,
        rs1: u8,
        imm: i16,
        op: ArithImmOp,
    },
    JumpAndLink {
        link_reg: u8,
        label: u32,
    },
    JumpAndLinkRegister {
        link_reg: u8,
        jump_reg: u8,
        label: u16,
    },
    Fence(u8, u8),
    FenceI,
    EnvCall,
    EnvBreak,
}

impl Instr {
    pub fn encode(self) -> Instruction {
        match self {
            Instr::CondBranch { diff, rs1, rs2, op } => {
                match op {
                    CondOp::Eq => instr_beq(rs1, rs2, diff),
                    CondOp::Ne => instr_bne(rs1, rs2, diff),
                    CondOp::Lt => instr_blt(rs1, rs2, diff),
                    CondOp::Ge => instr_bge(rs1, rs2, diff),
                    CondOp::Ltu => instr_bltu(rs1, rs2, diff),
                    CondOp::Geu => instr_bgeu(rs1, rs2, diff),
                }
            },
            Instr::Load { to, from_reg, from_imm, dtype } => {
                match dtype {
                    DataType::Byte => instr_lb(to, from_imm as u16, from_reg),
                    DataType::Half => instr_lh(to, from_imm as u16, from_reg),
                    DataType::Word => instr_lw(to, from_imm as u16, from_reg),
                    DataType::ByteUnsigned => instr_lbu(to, from_imm as u16, from_reg),
                    DataType::HalfUnsigned => instr_lhu(to, from_imm as u16, from_reg),
                }
            },
            Instr::Store { from, to_reg, to_imm, dtype } => {
                match dtype {
                    DataType::Byte | DataType::ByteUnsigned => instr_sb(from, to_imm as u16, to_reg),
                    DataType::Half | DataType::HalfUnsigned  => instr_sh(from, to_imm as u16, to_reg),
                    DataType::Word => instr_sw(from, to_imm as u16, to_reg),
                }
            },
            Instr::ArithmeticReg { rd, rs1, rs2, op } => {
                match op {
                    ArithOp::Add => instr_add(rd, rs1, rs2),
                    ArithOp::Sub => instr_sub(rd, rs1, rs2),
                    ArithOp::ShiftLeftLogic => instr_sll(rd, rs1, rs2),
                    ArithOp::ShiftRightLogic => instr_srl(rd, rs1, rs2),
                    ArithOp::ShiftRightArithmetic => instr_sra(rd, rs1, rs2),
                    ArithOp::SetLessThan => instr_slt(rd, rs1, rs2),
                    ArithOp::SetLessThanUnsigned => instr_sltu(rd, rs1, rs2),
                    ArithOp::Xor => instr_xor(rd, rs1, rs2),
                    ArithOp::Or => instr_or(rd, rs1, rs2),
                    ArithOp::And => instr_and(rd, rs1, rs2),
                }
            },
            Instr::ArithmeticImm { rd, rs1, imm, op } => {
                match op {
                    ArithImmOp::Add => instr_addi(rd, rs1, imm as u16),
                    ArithImmOp::ShiftLeftLogic => instr_slli(rd, rs1, imm as u8),
                    ArithImmOp::ShiftRightLogic => instr_srli(rd, rs1, imm as u8),
                    ArithImmOp::ShiftRightArithmetic => instr_srai(rd, rs1, imm as u8),
                    ArithImmOp::SetLessThan => instr_slt(rd, rs1, imm as u8),
                    ArithImmOp::SetLessThanUnsigned => instr_sltu(rd, rs1, imm as u8),
                    ArithImmOp::Xor => instr_xori(rd, rs1, imm as u16),
                    ArithImmOp::Or => instr_ori(rd, rs1, imm as u16),
                    ArithImmOp::And => instr_andi(rd, rs1, imm as u16),
                }
            },
            Instr::JumpAndLink { link_reg, label } => instr_jal(link_reg, label),
            Instr::JumpAndLinkRegister { link_reg, jump_reg, label } => instr_jalr(link_reg, jump_reg, label),
            Instr::Fence(pred, succ) => instr_fence(pred, succ),
            Instr::FenceI => instr_fence_i(),
            Instr::EnvCall => instr_ecall(),
            Instr::EnvBreak => instr_ebreak(),
        }
    }
}


#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Instruction(pub u32);

// For documentation on why this is like this, look here
// https://content.riscv.org/wp-content/uploads/2016/06/riscv-spec-v2.1.pdf page 54


//                          7       5           20      = 32
pub fn create_general(opcode: u8, rdimm: u8, rest: u32) -> Instruction {
    let data =  opcode as u32 & 0b01111111 |
        (rdimm as u32 & 0b00011111) << 7 |
        (rest & 0xFFFFF) << 12;

    Instruction(data)
}


// One destination register and two source registers (rd, rs1, rs2) ex: ADD
//                      7           5       3           5       5           7       = 32
pub fn create_type_r(opcode: u8, rd: u8, funct3: u8, rs1: u8, rs2: u8, funct7: u8) -> Instruction {
    let rest: u32 =
        funct3 as u32 & 0b00000111 |
            (rs1    as u32 & 0b00011111) << 3 |
            (rs2    as u32 & 0b00011111) << 8 |
            (funct7 as u32 & 0b01111111) << 13;
    create_general(opcode, rd, rest)
}

// One destination register and one source registers (rd, rs1) ex: ADDI
//                      7           5       3           5       12      = 32
pub fn create_type_i(opcode: u8, rd: u8, funct3: u8, rs1: u8, imm: u16) -> Instruction {
    let rest: u32 =
        funct3 as u32 & 0b00000111 |
            (rs1    as u32 & 0b00011111) << 3 |
            (imm    as u32 & 0x0FFF)     << 8;
    create_general(opcode, rd, rest)
}

// One two source registers and no destination registers (rs1, rs2) ex: store
//                      7           12       3           5       5       = 32
pub fn create_type_s(opcode: u8, imm: u16, funct3: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(opcode, imm as u8, funct3, rs1, rs2, (imm >> 5) as u8)
}

// One two source registers and no destination registers (rs1, rs2) ex: store
//                      7           12       3           5       5       = 32
pub fn create_type_sb(opcode: u8, imm: i16, funct3: u8, rs1: u8, rs2: u8) -> Instruction {
    let d11_07 = imm >> 10 & 1 | imm & 0b11110;// 5 bits
    let d31_25 = imm & 0b100000000000 | (imm & 0b0011111100000 << 1); // 7 bits
    create_type_r(opcode, d11_07 as u8, funct3, rs1, rs2, d31_25 as u8)
}

// Only one destination register (rd) and no minor opcode ex: LUI
//                      7           5       20      = 32
pub fn create_type_u(opcode: u8, rd: u8, imm: u32) -> Instruction {
    create_general(opcode, rd, imm >> 11)
}

// Only one destination register (rd) and no minor opcode ex: LUI
//                      7           5       20      = 32
pub fn create_type_uj(opcode: u8, rd: u8, imm: u32) -> Instruction {
    let imm2 =
        (imm >> 12) & 0x000FF | // [19..12] to [7..0]
            (imm >>  3) & 0x00100 | // [11..11] to [8..8]
            (imm <<  9) & 0xFFE00 | // [10.. 1] to [18..9]
            (imm <<  1) & 0x80000;  // [20..20] to [19..19]
    create_general(opcode, rd, imm2)
}

pub fn instr_lui(rd: u8, imm: u32) -> Instruction {
    create_type_u(0b0110111, rd, imm)
}

pub fn instr_auipc(rd: u8, imm: u32) -> Instruction {
    create_type_u(0b0010111, rd, imm)
}

pub fn instr_jal(rd: u8, imm: u32) -> Instruction {
    create_type_uj(0b1101111, rd, imm)
}

pub fn instr_jalr(rd: u8, rs1: u8, imm: u16) -> Instruction {
    create_type_i(0b1101111, rd, rs1, 0b000, imm)
}

pub fn instr_beq(rs1: u8, rs2: u8, imm: i16) -> Instruction {
    create_type_sb(0b1100011, imm, 0b000, rs1, rs2)
}

pub fn instr_bne(rs1: u8, rs2: u8, imm: i16) -> Instruction {
    create_type_sb(0b1100011, imm, 0b001, rs1, rs2)
}

pub fn instr_blt(rs1: u8, rs2: u8, imm: i16) -> Instruction {
    create_type_sb(0b1100011, imm, 0b100, rs1, rs2)
}

pub fn instr_bge(rs1: u8, rs2: u8, imm: i16) -> Instruction {
    create_type_sb(0b1100011, imm, 0b101, rs1, rs2)
}

pub fn instr_bltu(rs1: u8, rs2: u8, imm: i16) -> Instruction {
    create_type_sb(0b1100011, imm, 0b110, rs1, rs2)
}

pub fn instr_bgeu(rs1: u8, rs2: u8, imm: i16) -> Instruction {
    create_type_sb(0b1100011, imm, 0b111, rs1, rs2)
}

pub fn instr_lb(rd: u8, imm: u16, rs1: u8) -> Instruction {
    create_type_i(0b0000011, rd, 0b000, rs1, imm)
}

pub fn instr_lh(rd: u8, imm: u16, rs1: u8) -> Instruction {
    create_type_i(0b0000011, rd, 0b001, rs1, imm)
}

pub fn instr_lw(rd: u8, imm: u16, rs1: u8) -> Instruction {
    create_type_i(0b0000011, rd, 0b010, rs1, imm)
}

pub fn instr_lbu(rd: u8, imm: u16, rs1: u8) -> Instruction {
    create_type_i(0b0000011, rd, 0b100, rs1, imm)
}

pub fn instr_lhu(rd: u8, imm: u16, rs1: u8) -> Instruction {
    create_type_i(0b0000011, rd, 0b101, rs1, imm)
}

pub fn instr_sb(src: u8, imm: u16, base: u8) -> Instruction {
    create_type_s(0b0100011, imm, 0b000, base, src)
}

pub fn instr_sh(src: u8, imm: u16, base: u8) -> Instruction {
    create_type_s(0b0100011, imm, 0b001, base, src)
}

pub fn instr_sw(src: u8, imm: u16, base: u8) -> Instruction {
    create_type_s(0b0100011, imm, 0b010, base, src)
}

pub fn instr_addi(rd: u8, rs1: u8, imm: u16) -> Instruction {
    create_type_i(0b0010011, rd, 0b000, rs1, imm)
}

// Set less than immediate; rd = rs1 <= imm
pub fn instr_slti(rd: u8, rs1: u8, imm: u16) -> Instruction {
    create_type_i(0b0010011, rd, 0b010, rs1, imm)
}

pub fn instr_sltiu(rd: u8, rs1: u8, imm: u16) -> Instruction {
    create_type_i(0b0010011, rd, 0b011, rs1, imm)
}

pub fn instr_xori(rd: u8, rs1: u8, imm: u16) -> Instruction {
    create_type_i(0b0010011, rd, 0b100, rs1, imm)
}

pub fn instr_ori(rd: u8, rs1: u8, imm: u16) -> Instruction {
    create_type_i(0b0010011, rd, 0b110, rs1, imm)
}

pub fn instr_andi(rd: u8, rs1: u8, imm: u16) -> Instruction {
    create_type_i(0b0010011, rd, 0b111, rs1, imm)
}

pub fn instr_slli(rd: u8, rs1: u8, imm: u8) -> Instruction {
    create_type_i(0b0010011, rd, 0b001, rs1, (imm as u16) & 0b00011111)
}

pub fn instr_srli(rd: u8, rs1: u8, imm: u8) -> Instruction {
    create_type_i(0b0010011, rd, 0b101, rs1, (imm as u16) & 0b00011111)
}

pub fn instr_srai(rd: u8, rs1: u8, imm: u8) -> Instruction {
    create_type_i(0b0010011, rd, 0b101, rs1, (imm as u16) & 0b00011111 | 1 << 9)
}

pub fn instr_add(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b000, rs1, rs2, 0)
}

pub fn instr_sub(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b000, rs1, rs2, 1 << 5)
}

pub fn instr_sll(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b001, rs1, rs2, 0)
}

pub fn instr_slt(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b010, rs1, rs2, 0)
}

pub fn instr_sltu(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b011, rs1, rs2, 0)
}

pub fn instr_xor(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b100, rs1, rs2, 0)
}

pub fn instr_srl(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b101, rs1, rs2, 0)
}

pub fn instr_sra(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b101, rs1, rs2, 1 << 5)
}

pub fn instr_or(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b110, rs1, rs2, 0)
}

pub fn instr_and(rd: u8, rs1: u8, rs2: u8) -> Instruction {
    create_type_r(0b0110011, rd, 0b111, rs1, rs2, 0)
}

pub fn instr_fence(pred: u8, succ: u8) -> Instruction {
    create_type_i(0b0001111, 0, 0b000, 0, succ as u16 & 0b1111 | (pred as u16 & 0b1111) << 4)
}

pub fn instr_fence_i() -> Instruction {
    create_type_i(0b0001111, 0, 0b001, 0, 0)
}

pub fn instr_ecall() -> Instruction {
    create_type_i(0b1110011, 0, 0b000, 0, 0)
}

pub fn instr_ebreak() -> Instruction {
    create_type_i(0b1110011, 0, 0b000, 0, 1)
}

// CSR instructions

pub fn instr_csrrw(rd: u8, rs1: u8, csr: u16) -> Instruction {
    create_type_i(0b1110011, rd, 0b001, rs1, csr)
}

pub fn instr_csrrs(rd: u8, rs1: u8, csr: u16) -> Instruction {
    create_type_i(0b1110011, rd, 0b010, rs1, csr)
}

pub fn instr_csrrc(rd: u8, rs1: u8, csr: u16) -> Instruction {
    create_type_i(0b1110011, rd, 0b011, rs1, csr)
}

pub fn instr_csrrwi(rd: u8, zimm: u8, csr: u16) -> Instruction {
    create_type_i(0b1110011, rd, 0b101, zimm, csr)
}

pub fn instr_csrrsi(rd: u8, zimm: u8, csr: u16) -> Instruction {
    create_type_i(0b1110011, rd, 0b110, zimm, csr)
}

pub fn instr_csrrci(rd: u8, zimm: u8, csr: u16) -> Instruction {
    create_type_i(0b1110011, rd, 0b111, zimm, csr)
}

fn sign_extend(input: u32, bits: u8) -> i32 {
    let shift_bits = (mem::size_of::<u32>() * 8) as u32 - bits as u32;
    (input << shift_bits) as i32 >> shift_bits as i32
}


impl Instruction {
    // Decode
    pub fn to_str(self) -> String {
        let opcode = self.0 & 0b01111111;
        let rd = self.0 >> 7 & 0b00011111;
        let rest = self.0 >> 12;

        let funct3 = rest & 0b00000111;
        let rs1 = (rest >> 3) & 0b00011111;
        let rs2 = (rest >> 8) & 0b00011111;

        let funct7 = rest >> 13;

        let i_imm = sign_extend(rest >> 8, 12);
        let s_imm = sign_extend((rest >> 13) << 5 | rd, 12);
        let sb_imm = rd & 0b11110 |
            ((rest >> 12) & 0b111111) << 5 |
            rd & 1 << 11 |
            (rest >> 19) << 12;
        let u_imm = rest << 12;

        let rdn = register_name(rd as u8);
        let r1n = register_name(rs1 as u8);
        let r2n = register_name(rs2 as u8);

        match opcode {
            0b0110111 => {
                format!("LUI {}, {}", rdn, u_imm)
            },
            0b0010111 => {
                format!("AUIPC {}, {}", rdn, u_imm)
            },
            0b1101111 => {
                format!("JAL {}, {}", rdn, rest)
            },
            0b1100111 => {
                format!("JAL {}, {}", rdn, rest)
            },
            0b1100011 => {//Bxx
                match funct3 {
                    0b000 => format!("BEQ {}, {}, {}", r1n, r2n, sb_imm),
                    0b001 => format!("BNE {}, {}, {}", r1n, r2n, sb_imm),
                    0b100 => format!("BLT {}, {}, {}", r1n, r2n, sb_imm),
                    0b101 => format!("BGE {}, {}, {}", r1n, r2n, sb_imm),
                    0b110 => format!("BLTU {}, {}, {}", r1n, r2n, sb_imm),
                    0b111 => format!("BGEU {}, {}, {}", r1n, r2n, sb_imm),
                    _ => format!("Illegal branch instruction"),
                }
            },
            0b0000011 => {
                match funct3 {
                    0b000 => format!("LB {}, {}({})", rdn, i_imm, r1n),
                    0b001 => format!("LH {}, {}({})", rdn, i_imm, r1n),
                    0b010 => format!("LW {}, {}({})", rdn, i_imm, r1n),
                    0b100 => format!("LBU {}, {}({})", rdn, i_imm, r1n),
                    0b101 => format!("LHU {}, {}({})", rdn, i_imm, r1n),
                    _ => format!("Illegal load instruction"),
                }
            },
            0b0100011 => {
                match funct3 {
                    0b000 => format!("SB {}, {}({})", r2n, s_imm, r1n),
                    0b001 => format!("SH {}, {}({})", r2n, s_imm, r1n),
                    0b010 => format!("SW {}, {}({})", r2n, s_imm, r1n),
                    _ => format!("Illegal load instruction"),
                }
            }
            0b0010011 => {
                match funct3 {
                    0b000 => format!("ADDI {}, {}, {}",  rdn, r1n, i_imm),
                    0b010 => format!("SLTI {}, {}, {}",  rdn, r1n, i_imm),
                    0b011 => format!("SLTIU {}, {}, {}", rdn, r1n, i_imm),
                    0b100 => format!("XORI {}, {}, {}",  rdn, r1n, i_imm),
                    0b110 => format!("ORI {}, {}, {}",   rdn, r1n, i_imm),
                    0b111 => format!("ANDI {}, {}, {}",  rdn, r1n, i_imm),
                    0b001 => format!("SLLI {}, {}, {}",  rdn, r1n, r2n),
                    0b101 => {
                        let c = if funct7 != 0 { 'A' } else { 'L' };
                        format!("SR{}I {}, {}, {}", c, rdn, r1n, r2n)
                    },
                    _ => format!("Illegal immediate arithmetic instruction"),
                }
            },
            0b0110011 => {
                match funct3 {
                    0b000 => {
                        if funct7 == 0 {
                            format!("ADD {}, {}, {}", rdn, r1n, r2n)
                        } else {
                            format!("SUB {}, {}, {}", rdn, r1n, r2n)
                        }
                    },
                    0b001 => format!("SLL {}, {}, {}",  rdn, r1n, r2n),
                    0b010 => format!("SLT {}, {}, {}",  rdn, r1n, r2n),
                    0b011 => format!("SLTU {}, {}, {}", rdn, r1n, r2n),
                    0b100 => format!("XOR {}, {}, {}",  rdn, r1n, r2n),
                    0b101 => {
                        if funct7 == 0 {
                            format!("SRL {}, {}, {}", rdn, r1n, r2n)
                        } else {
                            format!("SRA {}, {}, {}", rdn, r1n, r2n)
                        }
                    },
                    0b110 => format!("OR {}, {}, {}", rdn, r1n, r2n),
                    0b111 => format!("AND {}, {}, {}", rdn, r1n, r2n),
                    _ => format!("Illegal arithmetic instruction"),
                }
            },
            0b0001111 => {
                match funct3 {
                    0b000 => {
                        let succ = i_imm & 0b1111;
                        let prev = (i_imm >> 4) & 0b1111;
                        format!("FENCE {}, {}", prev, succ)
                    },
                    0b001 => format!("FENCE.I"),
                    _ => format!("Illegal fence instruction")
                }
            }
            0b1110011 => {
                match funct3 {
                    0b000 => match i_imm {
                        0 => format!("ECALL"),
                        1 => format!("EBREAK"),
                        _ => format!("Illegal environment instruction"),
                    },
                    0b001 => format!("CSRRW {}, {}, {}", rdn, r1n, i_imm),
                    0b010 => format!("CSRRS {}, {}, {}", rdn, r1n, i_imm),
                    0b011 => format!("CSRRC {}, {}, {}", rdn, r1n, i_imm),
                    0b101 => format!("CSRRWI {}, {}, {}", rdn, r1n, i_imm),
                    0b110 => format!("CSRRSI {}, {}, {}", rdn, r1n, i_imm),
                    0b111 => format!("CSRRCI {}, {}, {}", rdn, r1n, i_imm),
                    _ => format!("Illegal CSR instruction"),
                }
            }
            _ => {
                format!("Illegal instruction")
            }
        }
    }
}

// ---------------------------------- REGISTERS ----------------------------------

pub fn register_return_address() -> u8 {
    return 1;
}

pub fn register_stack_pointer() -> u8 {
    return 2;
}

pub fn register_global_pointer() -> u8 {
    return 3;
}

pub fn register_thread_pointer() -> u8 {
    return 4;
}

pub fn register_argument(num: u8) -> Option<u8> {
    if num <= 8 { Some(num + 10) }
    else { None }
}

pub fn register_temporary(num: u8) -> Option<u8> {
    if num <= 2 { Some(num + 5) }
    else if num <= 6 { Some(num + 28) }
    else { None }
}

pub fn register_saved(num: u8) -> Option<u8> {
    if num <= 1 { Some(num + 8) }
    else if num <= 11 { Some(num + 18) }
    else { None }
}

pub fn register_name(num: u8) -> &'static str {
    let data: [&'static str; 32] = [
        "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3",
        "a4", "a5","a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        "t3", "t4", "t5", "t6",
    ];
    data.get(num as usize).unwrap_or(&"err")
}
