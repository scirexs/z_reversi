use std::{collections::BTreeMap, collections::HashSet, io, ops};
use anyhow::{bail, Result};
use itertools::Itertools;

type InputFunc = fn() -> String;

fn main() {
    let mut config = ConfigData::new();
    let mut play_data = PlayData::new();
    let mut next_dialog = Ok(DialogType::Title);

    loop {
        match next_dialog {
            Ok(x) => next_dialog = {
                match x {
                    DialogType::Title => show_standard(get_title()),
                    DialogType::Config => show_standard(get_config()),
                    DialogType::ConfigTurn => show_config_setter(get_config_turn(), &mut config),
                    DialogType::ConfigLevel => show_config_setter(get_config_level(), &mut config),
                    DialogType::ConfigAvail => show_config_setter(get_config_avail(), &mut config),
                    DialogType::Game => begin_game_env(get_game_env(), &mut play_data, &config),
                    DialogType::None => { break; },
                    _ => { println!("Unexpected dialog type on top level."); break; }
                }
            },
            Err(e) => { println!("{:?}", e); break; },
        }
    }
}

fn show_standard(df: DialogFrame) -> Result<DialogType> {
    Ok(df.begin()?.next)
}
fn show_config_setter(df: DialogFrame, config: &mut ConfigData) -> Result<DialogType> {
    let shift = df.begin()?;
    config.set_field(&shift.status);
    Ok(shift.next)
}
fn show_game_mover(df: DialogFrame, play_data: &mut PlayData) {
    if let Ok(shift) = df.begin() {
        play_data.set_field(&shift.status);
    }
}
fn begin_game_env(df: DialogFrame, play_data: &mut PlayData, config: &ConfigData) -> Result<DialogType> {
    play_data.init(config);
    let mut next_dialog = Ok(DialogType::from_turn(&play_data.first_turn));

    loop {
        match next_dialog? {
            DialogType::GamePlayer => show_game_mover(get_game_player(play_data), play_data),
            DialogType::GameNpc => show_game_mover(move_game_npc(play_data), play_data),
            DialogType::None => { break; }
            _ => { bail!("Unexpected dialog type in game env.") }
        };
        next_dialog = get_next_game_dialog(play_data);
    }
    Ok(df.begin()?.next)
}
fn get_next_game_dialog(play_data: &mut PlayData) -> Result<DialogType> {
    match play_data.turn {
        StoneType::Black => if let PlayTurn::First = play_data.first_turn { Ok(DialogType::GamePlayer) } else { Ok(DialogType::GameNpc) },
        StoneType::White => if let PlayTurn::First = play_data.first_turn { Ok(DialogType::GameNpc) } else { Ok(DialogType::GamePlayer) },
        StoneType::Null => { show_result(play_data); Ok(DialogType::None) },
    }
}
fn show_result(play_data: &PlayData) {
    let (you, npc) = match play_data.first_turn {
        PlayTurn::First => (play_data.get_stone_addrs(StoneType::Black).len(), play_data.get_stone_addrs(StoneType::White).len()),
        PlayTurn::Second => (play_data.get_stone_addrs(StoneType::White).len(), play_data.get_stone_addrs(StoneType::Black).len()),
        _ => (0, 0),
    };
    println!("{}", play_data.get_board_status());
    println!("You: {}, Npc: {}", you, npc);
    match you {
        you if you > npc => println!("You win!!!\n"),
        you if you < npc => println!("You lose...\n"),
        _ => println!("---Draw---\n")
    }
}
fn move_game_npc(play_data: &PlayData) -> DialogFrame {
    let address = match play_data.level {
        DifficultyLevel::Easy => get_move_easy(play_data),
        DifficultyLevel::Normal => get_move_normal(play_data),
    };
    get_game_npc(address)
}
fn get_move_easy(play_data: &PlayData) -> Address {
    let mut move_addr = Address(0, 0);
    let mut current_score = -200;
    for addr in play_data.availables.iter() {
        let eval_score = get_board_eval(&play_data.level, addr);
        if eval_score < current_score { continue; }
        if eval_score == current_score && get_random_bool() { continue; }
        current_score = eval_score;
        move_addr = *addr;
    }
    move_addr
}
fn get_move_normal(play_data: &PlayData) -> Address {
    if play_data.availables.len() == 1 { return *play_data.availables.first().unwrap_or(&Address(0, 0)); }
    let mut move_addr = Address(0, 0);
    let mut max_score = -200.;
    for addr in play_data.availables.iter() {
        let mut sim = play_data.light_clone();
        sim.sim_game(addr);
        let board_score = get_board_eval(&play_data.level, addr) as f64 - sim.availables.iter().map(|x| get_board_eval(&play_data.level, x)).max().unwrap_or_default() as f64;
        let stable_score = sim.get_stable_addrs(play_data.turn).len() as f64 - sim.get_stable_addrs(sim.turn).len() as f64;
        let avail_score = sim.availables.len() as f64;
        let total_score = scale_stable_score(stable_score) * 0.6 + scale_board_score(board_score) * 0.3 - scale_avail_score(avail_score) * 0.1;
        if total_score > max_score { max_score = total_score; move_addr = *addr; }
    }
    move_addr
}
fn get_board_eval(level: &DifficultyLevel, addr: &Address) -> i32 {
    const EVAL_BOARD_EASY: [[i32; 8]; 8] = [
        [120, -20, 20,  5,  5, 20, -20, 120],
        [-20, -40, -5, -5, -5, -5, -40, -20],
        [ 20,  -5, 15,  3,  3, 15,  -5,  20],
        [  5,  -5,  3,  3,  3,  3,  -5,   5],
        [  5,  -5,  3,  3,  3,  3,  -5,   5],
        [ 20,  -5, 15,  3,  3, 15,  -5,  20],
        [-20, -40, -5, -5, -5, -5, -40, -20],
        [120, -20, 20,  5,  5, 20, -20, 120]];
    const EVAL_BOARD_NORMAL: [[i32; 8]; 8] = [
        [100, -40, 20,  5,  5, 20, -40, 100],
        [-40, -80, -1, -1, -1, -1, -80, -40],
        [ 20,  -1,  5,  1,  1,  5,  -1,  20],
        [  5,  -1,  1,  0,  0,  1,  -1,   5],
        [  5,  -1,  1,  0,  0,  1,  -1,   5],
        [ 20,  -1,  5,  1,  1,  5,  -1,  20],
        [-40, -80, -1, -1, -1, -1, -80, -40],
        [100, -40, 20,  5,  5, 20, -40, 100]];
    let Address(r, c) = *addr;
    match level {
        DifficultyLevel::Easy => EVAL_BOARD_EASY[r][c],
        DifficultyLevel::Normal => EVAL_BOARD_NORMAL[r][c],
    }
}
fn scale_stable_score(score: f64) -> f64 { score / 16. }
fn scale_board_score(score: f64) -> f64 { score / 100. }
fn scale_avail_score(score: f64) -> f64 {
    let axis = 6.;
    let raw = score - axis;
    let scale = raw.signum() * (raw.powi(2) / axis.powi(2));
    if scale < 1. { scale } else { 1. }
}


///////////////////////////////////////////////////////////////////////
#[derive(Clone, Copy)]
enum DialogType {
    None,
    Title,
    Config,
    ConfigTurn,
    ConfigLevel,
    ConfigAvail,
    Game,
    GamePlayer,
    GameNpc,
}
impl DialogType {
    fn from_turn(turn: &PlayTurn) -> Self {
        match turn {
            PlayTurn::First => DialogType::GamePlayer,
            PlayTurn::Second => DialogType::GameNpc,
            _ => DialogType::None,
        }
    }
}
#[derive(Clone)]
enum DataEnum {
    None,
    Turn(PlayTurn),
    Level(DifficultyLevel),
    Avail(bool),
    Move(Address),
}
#[derive(Clone, Copy, Debug)]
enum PlayTurn {
    First,
    Second,
    Random,
}
impl PlayTurn {
    fn decide_turn(&self) -> Self {
        match self {
            PlayTurn::Random => if get_random_bool() { PlayTurn::First } else { PlayTurn::Second },
            PlayTurn::First => PlayTurn::First,
            PlayTurn::Second => PlayTurn::Second,
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum DifficultyLevel {
    Easy,
    Normal,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum StoneType {
    Null,
    Black,
    White,
}
impl StoneType {
    fn toggle_stone(&self) -> Self {
        match self {
            StoneType::Black => StoneType::White,
            StoneType::White => StoneType::Black,
            _ => StoneType::Null,
        }
    }
    fn get_string(&self) -> &str {
        match self {
            StoneType::Null => "-",
            StoneType::Black => "x",
            StoneType::White => "o",
        }
    }
    fn get_another_string(&self) -> &str {
        match self {
            StoneType::Null => "?",
            StoneType::Black => "X",
            StoneType::White => "O",
        }
    }
}
trait SetData {
    fn set_field(&mut self, value: &DataEnum);
}
#[derive(Debug)]
struct ConfigData {
    turn: PlayTurn,
    level: DifficultyLevel,
    avail: bool,
}
impl ConfigData {
    fn new() -> Self {
        Self { turn: PlayTurn::Random, level: DifficultyLevel::Normal, avail: false }
    }
}
impl SetData for ConfigData {
    fn set_field(&mut self, value: &DataEnum) {
        match value {
            DataEnum::Turn(x) => self.turn = *x,
            DataEnum::Level(x) => self.level = *x,
            DataEnum::Avail(x) => self.avail = *x,
            _ => ()
        }
    }
}
struct PlayData {
    first_turn: PlayTurn,
    level: DifficultyLevel,
    disp_avail: bool,
    turn: StoneType,
    board: [[StoneType; 8]; 8],
    availables: Vec<Address>,
    history: Vec<(StoneType, Address)>,
}
impl SetData for PlayData {
    fn set_field(&mut self, value: &DataEnum) {
        if let DataEnum::Move(x) = value { self.move_game(x); }
    }
}
impl PlayData {
    const PLAIN_BOARD:[[StoneType; 8]; 8] = [[StoneType::Null; 8]; 8];
    fn new() -> Self {
        Self {
            first_turn: PlayTurn::Random,
            level: DifficultyLevel::Normal,
            disp_avail: false,
            turn: StoneType::Black,
            board: Self::PLAIN_BOARD,
            availables: Vec::new(),
            history: Vec::new(),
        }
    }
    fn init(&mut self, config: &ConfigData) {
        self.first_turn = config.turn.decide_turn();
        self.level = config.level;
        self.disp_avail = config.avail;
        self.turn = StoneType::Black;
        self.init_stone();
        self.history = Vec::new();
        self.availables = self.get_availables();
    }
    fn init_stone(&mut self) {
        self.board = Self::PLAIN_BOARD;
        self.board[3][3] = StoneType::White;
        self.board[4][3] = StoneType::Black;
        self.board[3][4] = StoneType::Black;
        self.board[4][4] = StoneType::White;
    }
    fn light_clone(&self) -> Self {
        Self {
            first_turn: self.first_turn,
            level: self.level,
            disp_avail: self.disp_avail,
            turn: self.turn,
            board: self.board,
            availables: self.availables.clone(),
            history: Vec::new(),
        }
    }
    fn get_board_addrs() -> Vec<Address> {
        let mut addrs = Vec::new();
        for rc in (0..Self::PLAIN_BOARD.len()).cartesian_product(0..Self::PLAIN_BOARD.len()) {
            addrs.push(Address::new(rc));
        }
        addrs
    }
    fn get_around_offsets() -> Vec<Offset> {
        let mut offsets = Vec::new();
        for rc in (-1..=1).cartesian_product(-1..=1) {
            let offset = Offset::new(rc);
            if !offset.is_zero() { offsets.push(offset) }
        }
        offsets
    }
    fn get_offset_vec_addrs(source: &Address, offset: Offset) -> Vec<Address> {
        let mut addrs = Vec::new();
        if offset.is_zero() { return addrs; }
        for i in 0..(Self::PLAIN_BOARD.len() as i32) {
            if let Ok(addr) = *source + offset * i {
                addrs.push(addr);
            } else { return addrs; }
        }
        addrs
    }
    fn get_stone(&self, addr: &Address) -> StoneType {
        let Address(r, c) = *addr;
        self.board[r][c]
    }
    fn set_stone(&mut self, addr: &Address, stone: StoneType) {
        let Address(r, c) = *addr;
        self.board[r][c] = stone;
    }
    fn move_game(&mut self, addr: &Address) -> bool {
        if !self.availables.contains(addr) { return false; }
        self.move_stone(addr);
        self.history.push((self.turn, *addr));
        self.pass_turn();
        if !self.availables.len() > 0 { self.pass_turn() }
        if !self.availables.len() > 0 { self.turn = StoneType::Null }
        true
    }
    fn sim_game(&mut self, addr: &Address) -> bool {
        if !self.availables.contains(addr) { return false; }
        self.move_stone(addr);
        self.pass_turn();
        true
    }
    fn move_stone(&mut self, source: &Address) {
        self.set_stone(source, self.turn);
        for offset in Self::get_around_offsets() {
            for addr in self.get_reverse_addrs(source, offset) {
                self.set_stone(&addr, self.turn);
            }
        }
    }
    fn pass_turn(&mut self) {
        self.turn = self.turn.toggle_stone();
        self.availables = self.get_availables();
    }
    fn get_stone_addrs(&self, stone: StoneType) -> Vec<Address> {
        Self::get_board_addrs().into_iter().filter(|x| self.get_stone(x) == stone).collect()
    }
    fn get_board_status(&self) -> String {
        const HEADER: &str = "  _a_b_c_d_e_f_g_h_\n";
        const FOOTER: &str = "  -a-b-c-d-e-f-g-h-\n";
        let mut status = String::with_capacity(220);
        let mut row_count = 1;
        status += HEADER;
        for addr in Self::get_board_addrs() {
            if addr.is_left_edge() { status += &format!("{}| ", row_count); }
            let stone = self.get_stone(&addr);
            status += if self.is_another_string(&addr, stone) { stone.get_another_string() } else { stone.get_string() };
            status += " ";
            if addr.is_right_edge() { status += &format!("|{}\n", row_count); row_count += 1; }
        }
        status += FOOTER;
        status
    }
    fn is_another_string(&self, addr: &Address, stone: StoneType) -> bool {
        match stone {
            StoneType::Null => self.disp_avail && self.availables.contains(addr),
            _ => if let Some((_, last_addr)) = self.history.last() { last_addr == addr } else { false },
        }
    }
    fn get_availables(&self) -> Vec<Address> {
        self.get_stone_addrs(StoneType::Null)
            .into_iter()
            .filter(|addr| self.is_available(addr))
            .collect()
    }
    fn is_available(&self, addr: &Address) -> bool {
        for offset in Self::get_around_offsets() {
            if !self.get_reverse_addrs(addr, offset).is_empty() { return true; }
        }
        false
    }
    fn get_reverse_addrs(&self, source: &Address, offset: Offset) -> Vec<Address> {
        const EMPTY: Vec<Address> = Vec::new();
        let mut addrs = Vec::new();
        for addr in Self::get_offset_vec_addrs(source, offset) {
            if addr == *source { continue; }
            let stone = self.get_stone(&addr);
            if stone == StoneType::Null {
                return EMPTY;
            } else if stone == self.turn {
                return addrs;
            } else if stone == self.turn.toggle_stone() {
                addrs.push(addr);
            }
        }
        EMPTY
    }
    fn get_stable_addrs(&self, stone: StoneType) -> HashSet<Address> {
        let mut addrs = HashSet::new();
        for corner in Address::get_corners() {
            let (edge_addrs, edge_offsets) = self.get_edge_stable_addrs(&corner, stone);
            for (base, offset) in edge_addrs.into_iter().zip(edge_offsets) {
                addrs.extend(self.get_inner_stable_addrs(base, Self::get_inner_scan_offset(&corner, &offset), stone));
            }
        }
        addrs
    }
    fn get_edge_stable_addrs(&self, source: &Address, stone: StoneType) -> (Vec<Vec<Address>>, Vec<Offset>) {
        let mut offsets = Vec::new();
        let mut addrs = Vec::new();
        for offset in Self::get_edge_scan_offsets(source).into_iter() {
            offsets.push(offset);
            addrs.push(self.get_consecutive_addrs(source, offset, stone));
        }
        (addrs, offsets)
    }
    fn get_inner_stable_addrs(&self, edge_addrs: Vec<Address>, offset_diagonal: Offset, stone: StoneType) -> Vec<Address> {
        let mut addrs = Vec::new();
        let mut flawless_count = 100;
        for base in edge_addrs {
            for (i, addr) in Self::get_offset_vec_addrs(&base, offset_diagonal).into_iter().enumerate() {
                if self.get_stone(&addr) != stone { flawless_count = i - 1; break; }
                if i > flawless_count { break; }
                addrs.push(addr);
            }
        }
        addrs
    }
    fn get_consecutive_addrs(&self, source: &Address, offset: Offset, stone: StoneType) -> Vec<Address> {
        let mut addrs = Vec::new();
        for addr in Self::get_offset_vec_addrs(source, offset).into_iter() {
            if self.get_stone(&addr) != stone { break; }
            addrs.push(addr);
        }
        addrs
    }
    fn get_edge_scan_offsets(source: &Address) -> Vec<Offset> {
        match source {
            addr if addr.is_up_left_corner() => vec![Offset::to_down(), Offset::to_right()],
            addr if addr.is_up_right_corner() => vec![Offset::to_down(), Offset::to_left()],
            addr if addr.is_down_left_corner() => vec![Offset::to_up(), Offset::to_right()],
            addr if addr.is_down_right_corner() => vec![Offset::to_up(), Offset::to_left()],
            _ => Vec::new()
        }
    }
    fn get_inner_scan_offset(source: &Address, offset: &Offset) -> Offset {
        match source {
            addr if addr.is_up_left_corner() && offset.is_down() => Offset::to_up_right(),
            addr if addr.is_up_left_corner() && offset.is_right() => Offset::to_down_left(),
            addr if addr.is_up_right_corner() && offset.is_down() => Offset::to_up_left(),
            addr if addr.is_up_right_corner() && offset.is_left() => Offset::to_down_right(),
            addr if addr.is_down_left_corner() && offset.is_up() => Offset::to_down_right(),
            addr if addr.is_down_left_corner() && offset.is_right() => Offset::to_up_left(),
            addr if addr.is_down_right_corner() && offset.is_up() => Offset::to_down_left(),
            addr if addr.is_down_right_corner() && offset.is_left() => Offset::to_up_right(),
            _ => Offset::zero()
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Address(usize, usize);
impl Address {
    const ADDR_COLS: [char; 8] = ['a','b','c','d','e','f','g','h'];
    const ADDR_ROWS: [char; 8] = ['1','2','3','4','5','6','7','8'];
    fn new(rc: (usize, usize)) -> Self { Self(rc.0, rc.1) }
    fn max_row() -> usize { Self::ADDR_ROWS.len()-1 }
    fn max_col() -> usize { Self::ADDR_COLS.len()-1 }
    // fn from_string(s: &String) -> Option<Self> {
    //     if let Some((r, c)) = Self::get_first_2chars(s) {
    //         if Self::is_valid_str_col(&c) && Self::is_valid_str_row(&r) {
    //             Some(Address((r as u8 - '1' as u8) as usize, (c as u8 - 'a' as u8) as usize))
    //         } else { None }
    //     } else { None }
    // }
    // fn get_first_2chars(s: &String) -> Option<(char, char)> {
    //     if s.chars().count() != 2 { return None; }
    //     Some((s.chars().nth(0).unwrap_or_default(), s.chars().nth(1).unwrap_or_default()))
    // }
    // fn is_valid_str_col(c: &char) -> bool { !Self::ADDR_COLS.contains(c) }
    // fn is_valid_str_row(r: &char) -> bool { !Self::ADDR_ROWS.contains(r) }
    // fn get_i32_addr(&self) -> (i32, i32) { (self.0 as i32, self.1 as i32) }
    fn get_string(&self) -> String {
        if !self.is_valid_address() { return String::new(); }
        let Address(r, c) = *self;
        format!("{}{}", Self::ADDR_COLS[c], Self::ADDR_ROWS[r])
    }
    fn is_valid_address(&self) -> bool { self.0 < Self::ADDR_ROWS.len() && self.1 < Self::ADDR_COLS.len() }
    fn get_up_left_corner() -> Self { Address(0, 0) }
    fn get_up_right_corner() -> Self { Address(0, Self::max_col()) }
    fn get_down_left_corner() -> Self { Address(Self::max_row(), 0) }
    fn get_down_right_corner() -> Self { Address(Self::max_row(), Self::max_col()) }
    fn is_up_left_corner(&self) -> bool { *self == Self::get_up_left_corner() }
    fn is_up_right_corner(&self) -> bool { *self == Self::get_up_right_corner() }
    fn is_down_left_corner(&self) -> bool { *self == Self::get_down_left_corner() }
    fn is_down_right_corner(&self) -> bool { *self == Self::get_down_right_corner() }
    fn get_corners() -> Vec<Address> {
        vec![Self::get_up_left_corner(),Self::get_up_right_corner(),Self::get_down_left_corner(),Self::get_down_right_corner()]
    }
    // fn is_up_edge(&self) -> bool { self.0 == 0 }
    // fn is_down_edge(&self) -> bool { self.0 == Self::max_row() }
    fn is_left_edge(&self) -> bool { self.1 == 0 }
    fn is_right_edge(&self) -> bool { self.1 == Self::max_col()}
}
impl ops::Add<Offset> for Address {
    type Output = Result<Address>;
    fn add(self, rhs: Offset) -> Self::Output {
        let addr = Address(usize::try_from((self.0 as i32) + rhs.0)?, usize::try_from((self.1 as i32) + rhs.1)?);
        if addr.is_valid_address() { Ok(addr) } else { bail!("Add offset error in address.") }
    }
}
#[derive(Clone, Copy, PartialEq, Default, Debug)]
struct Offset(i32, i32);
impl Offset {
    fn new(rc: (i32, i32)) -> Self { Self(rc.0, rc.1) }
    fn zero() -> Self { Self(0, 0) }
    fn is_zero(&self) -> bool { self.0 == 0 && self.1 == 0 }
    fn is_up(&self) -> bool { self.0 == -1 && self.1 == 0 }
    fn is_down(&self) -> bool { self.0 == 1 && self.1 == 0 }
    fn is_left(&self) -> bool { self.0 == 0 && self.1 == -1 }
    fn is_right(&self) -> bool { self.0 == 0 && self.1 == 1 }
    fn to_up() -> Self { Self(-1, 0) }
    fn to_down() -> Self { Self(1, 0) }
    fn to_left() -> Self { Self(0, -1) }
    fn to_right() -> Self { Self(0, 1) }
    fn to_up_left() -> Self { Self(-1, -1) }
    fn to_up_right() -> Self { Self(-1, 1) }
    fn to_down_left() -> Self { Self(1, -1) }
    fn to_down_right() -> Self { Self(1, 1) }
}
impl ops::Mul<i32> for Offset {
    type Output = Offset;
    fn mul(self, rhs: i32) -> Self::Output {
        Offset(self.0 * rhs, self.1 * rhs)
    }
}

#[derive(Clone)]
struct StatusShift {
    status: DataEnum,
    next: DialogType,
}
impl StatusShift {
    fn new(status: DataEnum, next: DialogType) -> Self {
        Self { status, next }
    }
}
struct ActionSet {
    help: String,
    resp: String,
    action: StatusShift,
}
impl ActionSet {
    fn new(help: String, resp: String, status: DataEnum, next: DialogType) -> Self {
        Self { help, resp, action: StatusShift::new(status, next) }
    }
}

struct DialogFrame {
    info_text: String,
    action_map: BTreeMap<String, ActionSet>,
    input_func: InputFunc,
}
impl DialogFrame {
    fn new<T: Into<String>>(info: T, action_preset: Vec<(T, T, T, DataEnum, DialogType)>, input_func: InputFunc) -> Self {
        let mut action_map = BTreeMap::new();
        for (key, help, resp, status, next) in action_preset {
            action_map.insert(key.into(), ActionSet::new(help.into(), resp.into(), status, next));
        }
        Self { info_text: info.into(), action_map, input_func }
    }
    fn show_info(&self) {
        if !self.info_text.is_empty() {
            println!();
            println!("{}", self.info_text);
            self.action_map.keys()
                .filter_map(|k| self.action_map.get_key_value(k))
                .for_each(|(x, y)| if !y.help.is_empty() {println!("{} -> {}", *x, y.help);});
            println!();
        }
    }
    fn show_invalid(&self) {
        println!("Not acceptable string.");
        println!("availables: [{}]", self.action_map.keys().map(|s| &**s).collect::<Vec<_>>().join(","));
        println!();
    }
    fn begin(&self) -> Result<StatusShift> {
        if self.action_map.is_empty() { bail!("Not defined action set."); }
        self.show_info();
        loop {
            match self.action_map.get(&(self.input_func)()) {
                None => (),
                Some(act_set) => {
                    if !act_set.resp.is_empty() { println!("{}\n", act_set.resp); }
                    return Ok(act_set.action.clone());
                },
            }
            self.show_invalid();
        }
    }
}

fn get_input() -> String {
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_ok() {
        input = input.trim().to_string()
    }
    input
}
fn auto_input() -> String {
    "auto".to_string()
}
fn get_random_bool() -> bool {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() % 2 > 0
}


///////////////////////////////////////////////////////////////////////
fn get_title() -> DialogFrame {
    DialogFrame::new(
        "/***** R E V E R S I *****/\nPlease type char.",
        vec![
            ("1", "begin game", "Let's begin game!", DataEnum::None, DialogType::Game),
            ("2", "set config", "", DataEnum::None, DialogType::Config),
            ("3", "quit game", "Quit game, bye.", DataEnum::None, DialogType::None),
        ],
        get_input as InputFunc,
    )
}
fn get_config() -> DialogFrame {
    DialogFrame::new(
        "Select the item to be changed.",
        vec![
            ("1", "player turn", "", DataEnum::None, DialogType::ConfigTurn),
            ("2", "level of NPC", "", DataEnum::None, DialogType::ConfigLevel),
            ("3", "visibility of next move", "", DataEnum::None, DialogType::ConfigAvail),
            ("4", "return to title (cancel)", "", DataEnum::None, DialogType::Title),
        ],
        get_input as InputFunc,
    )
}
fn get_config_turn() -> DialogFrame {
    DialogFrame::new(
        "Choose your turn.",
        vec![
            ("1", "first (x)", "Set your turn as first. (x)", DataEnum::Turn(PlayTurn::First), DialogType::Title),
            ("2", "second (o)", "Set your turn as second. (o)", DataEnum::Turn(PlayTurn::Second), DialogType::Title),
            ("3", "random", "Set your turn as random.", DataEnum::Turn(PlayTurn::Random), DialogType::Title),
        ],
        get_input as InputFunc,
    )
}
fn get_config_level() -> DialogFrame {
    DialogFrame::new(
        "Choose NPC level.",
        vec![
            ("1", "easy", "Set NPC level to easy.", DataEnum::Level(DifficultyLevel::Easy), DialogType::Title),
            ("2", "normal", "Set NPC level to normal.", DataEnum::Level(DifficultyLevel::Normal), DialogType::Title),
        ],
        get_input as InputFunc,
    )
}
fn get_config_avail() -> DialogFrame {
    DialogFrame::new(
        "Choose display availability.",
        vec![
            ("1", "on", "Show available addresses.", DataEnum::Avail(true), DialogType::Title),
            ("2", "off", "Hide availabilities.", DataEnum::Avail(false), DialogType::Title),
        ],
        get_input as InputFunc,
    )
}
fn get_game_env() -> DialogFrame {
    DialogFrame::new(
        "",
        vec![
            ("auto", "end of env, move to title", "", DataEnum::None, DialogType::Title),
        ],
        auto_input as InputFunc,
    )
}
fn get_game_player(play_data: &PlayData) -> DialogFrame {
    DialogFrame::new(
        play_data.get_board_status(),
        get_player_preset(play_data),
        get_input as InputFunc,
    )
}
fn get_player_preset(play_data: &PlayData) -> Vec<(String, String, String, DataEnum, DialogType)> {
    let mut preset = Vec::new();
    for addr in play_data.availables.iter() {
        preset.push((addr.get_string(), "".to_string(), "".to_string(), DataEnum::Move(*addr), DialogType::GameNpc));
    }
    preset
}
fn get_game_npc(addr: Address) -> DialogFrame {
    DialogFrame::new(
        "",
        vec![
            ("auto", "", "", DataEnum::Move(addr), DialogType::GamePlayer)
        ],
        auto_input as InputFunc,
    )
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_offset() {
        let ofst_zero = Offset::zero();
        let ofst_big = Offset::new((-5, -5));
        let ofst_up = Offset(-1, 0);
        let ofst_down = Offset(1, 0);
        let ofst_left = Offset(0, -1);
        let ofst_right = Offset(0, 1);
        let ofst_up_left = Offset(-1, -1);
        let ofst_up_right = Offset(-1, 1);
        let ofst_down_left = Offset(1, -1);
        let ofst_down_right = Offset(1, 1);

        assert!(ofst_zero.is_zero());
        assert!(!ofst_up.is_zero());
        assert!(ofst_up.is_up());
        assert!(!ofst_down.is_up());
        assert!(!ofst_up.is_down());
        assert!(ofst_down.is_down());
        assert!(ofst_left.is_left());
        assert!(!ofst_right.is_left());
        assert!(!ofst_left.is_right());
        assert!(ofst_right.is_right());

        assert_eq!(ofst_up, Offset::to_up());
        assert_eq!(ofst_down, Offset::to_down());
        assert_eq!(ofst_left, Offset::to_left());
        assert_eq!(ofst_right, Offset::to_right());
        assert_eq!(ofst_up_left, Offset::to_up_left());
        assert_eq!(ofst_up_right, Offset::to_up_right());
        assert_eq!(ofst_down_left, Offset::to_down_left());
        assert_eq!(ofst_down_right, Offset::to_down_right());

        assert_eq!(ofst_big, ofst_up_left * 5);
    }
    #[test]
    fn test_address() {
        assert_eq!(7, Address::max_row());
        assert_eq!(7, Address::max_row());

        let addr_up_left = Address::get_up_left_corner();
        let addr_up_right = Address::get_up_right_corner();
        let addr_down_left = Address::get_down_left_corner();
        let addr_down_right = Address::get_down_right_corner();
        let addr_mid = Address(4, 4);
        let addr_up_mid = Address(0, 3);
        let addr_down_mid = Address(7, 4);
        let addr_left_mid = Address(4, 0);
        let addr_right_mid = Address(3, 7);

        assert!(addr_up_left.is_up_left_corner());
        assert!(addr_up_right.is_up_right_corner());
        assert!(addr_down_left.is_down_left_corner());
        assert!(addr_down_right.is_down_right_corner());

        assert!(addr_up_left.is_valid_address());
        assert!(addr_up_right.is_valid_address());
        assert!(addr_down_left.is_valid_address());
        assert!(addr_down_right.is_valid_address());
        assert!(addr_mid.is_valid_address());
        assert!(addr_up_mid.is_valid_address());
        assert!(addr_down_mid.is_valid_address());
        assert!(addr_left_mid.is_valid_address());
        assert!(addr_right_mid.is_valid_address());
        assert!(!Address(8, 8).is_valid_address());
        assert!(!Address(0, 8).is_valid_address());
        assert!(!Address(8, 0).is_valid_address());

        assert!(addr_up_left.is_left_edge());
        assert!(!addr_up_right.is_left_edge());
        assert!(addr_down_left.is_left_edge());
        assert!(!addr_down_right.is_left_edge());
        assert!(!addr_mid.is_left_edge());
        assert!(!addr_up_mid.is_left_edge());
        assert!(!addr_down_mid.is_left_edge());
        assert!(addr_left_mid.is_left_edge());
        assert!(!addr_right_mid.is_left_edge());

        assert!(!addr_up_left.is_right_edge());
        assert!(addr_up_right.is_right_edge());
        assert!(!addr_down_left.is_right_edge());
        assert!(addr_down_right.is_right_edge());
        assert!(!addr_mid.is_right_edge());
        assert!(!addr_up_mid.is_right_edge());
        assert!(!addr_down_mid.is_right_edge());
        assert!(!addr_left_mid.is_right_edge());
        assert!(addr_right_mid.is_right_edge());

        assert_eq!(vec![addr_up_left, addr_up_right, addr_down_left, addr_down_right], Address::get_corners());

        assert_eq!(addr_up_left.get_string(),"a1".to_string());
        assert_eq!(addr_up_right.get_string(),"h1".to_string());
        assert_eq!(addr_down_left.get_string(),"a8".to_string());
        assert_eq!(addr_down_right.get_string(),"h8".to_string());
        assert_eq!(addr_mid.get_string(),"e5".to_string());
        assert_eq!(addr_up_mid.get_string(),"d1".to_string());
        assert_eq!(addr_down_mid.get_string(),"e8".to_string());
        assert_eq!(addr_left_mid.get_string(),"a5".to_string());
        assert_eq!(addr_right_mid.get_string(),"h4".to_string());

        assert_eq!(addr_up_right, (addr_up_left + Offset::to_right() * 7).unwrap());
    }

    #[test]
    fn test_run1() {
        let config = ConfigData{ level: DifficultyLevel::Easy, turn: PlayTurn::Second, avail: true };
        let mut play_data = PlayData::new();
        play_data.init(&config);
        println!("{}", play_data.get_board_status());
        let addr = Address(3, 2);
        play_data.move_game(&addr);
        println!("{}", play_data.get_board_status());
        let addr = Address(2, 2);
        play_data.move_game(&addr);
        println!("{}", play_data.get_board_status());
        let addr = Address(2, 3);
        play_data.move_game(&addr);
        println!("{}", play_data.get_board_status());
        let addr = Address(2, 4);
        play_data.move_game(&addr);
        println!("{}", play_data.get_board_status());
        show_result(&play_data);
    }
}