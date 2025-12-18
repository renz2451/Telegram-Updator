#!/usr/bin/env python3
"""
TELEGRAM OFFSET UPDATER BOT FOR .CS DUMP FILES
Complete version with automatic file reading and processing
"""

import os
import re
import json
import shutil
import asyncio
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Telegram Bot
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

# ============ CONFIGURATION ============
# Load from environment or config file
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
ALLOWED_USER_IDS = [int(x) for x in os.environ.get("ALLOWED_USER_IDS", "5682792112,6064653643").split(",")]

# File upload limits (in bytes)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_SOURCE_EXTENSIONS = ['.cpp', '.c', '.h', '.hpp', '.cc', '.cxx', '.txt']
ALLOWED_DUMP_EXTENSIONS = ['.cs', '.txt', '.log', '.dump']

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============ DATA STRUCTURES ============
class UserSession:
    """Store user session data"""
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.source_file: Optional[Path] = None
        self.old_dump: Optional[Path] = None
        self.new_dump: Optional[Path] = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"user_{user_id}_"))
        self.status = "awaiting_files"
        self.last_activity = datetime.now()
        
        # Create subdirectories
        (self.temp_dir / "uploads").mkdir(exist_ok=True)
        (self.temp_dir / "outputs").mkdir(exist_ok=True)
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned temp dir for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error cleaning temp dir: {e}")
    
    def is_ready(self) -> bool:
        """Check if all required files are uploaded"""
        return all([self.source_file, self.old_dump, self.new_dump])
    
    def get_status_text(self) -> str:
        """Get formatted status text"""
        files = [
            f"{'✅' if self.source_file else '❌'} Source: {self.source_file.name if self.source_file else 'Not uploaded'}",
            f"{'✅' if self.old_dump else '❌'} Old Dump: {self.old_dump.name if self.old_dump else 'Not uploaded'}",
            f"{'✅' if self.new_dump else '❌'} New Dump: {self.new_dump.name if self.new_dump else 'Not uploaded'}"
        ]
        return "\n".join(files)

# Global session storage
user_sessions: Dict[int, UserSession] = {}

# ============ KEYBOARD HELPERS ============
def get_main_keyboard() -> InlineKeyboardMarkup:
    """Main menu keyboard"""
    buttons = [
        [InlineKeyboardButton("📤 Upload Files", callback_data="upload")],
        [InlineKeyboardButton("🔄 Check Status", callback_data="status")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
        [
            InlineKeyboardButton("❓ Help", callback_data="help"),
            InlineKeyboardButton("🗑️ Clear All", callback_data="clear")
        ]
    ]
    return InlineKeyboardMarkup(buttons)

def get_file_type_keyboard() -> InlineKeyboardMarkup:
    """Keyboard for file type selection"""
    buttons = [
        [InlineKeyboardButton("📄 C++ Source File", callback_data="type_source")],
        [InlineKeyboardButton("📁 OLD .cs Dump", callback_data="type_old")],
        [InlineKeyboardButton("📁 NEW .cs Dump", callback_data="type_new")],
        [InlineKeyboardButton("🔙 Back", callback_data="back")]
    ]
    return InlineKeyboardMarkup(buttons)

def get_processing_keyboard() -> InlineKeyboardMarkup:
    """Keyboard during processing"""
    buttons = [
        [InlineKeyboardButton("⏸️ Pause", callback_data="pause")],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(buttons)

# ============ COMMAND HANDLERS ============
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    user_id = update.effective_user.id
    
    if user_id not in ALLOWED_USER_IDS:
        await update.message.reply_text("❌ You are not authorized to use this bot.")
        return
    
    # Initialize or refresh session
    if user_id in user_sessions:
        user_sessions[user_id].cleanup()
    
    user_sessions[user_id] = UserSession(user_id)
    
    welcome_text = """
🤖 *Offset Updater Bot*

*Automatically update offsets in your C++ files using .cs dump files!*

📁 *Required Files:*
1. `source.cpp` - Your C++ source code with offsets
2. `old_dump.cs` - OLD C# dump with function addresses
3. `new_dump.cs` - NEW C# dump with updated addresses

🔄 *How it works:*
1. Send me all 3 files
2. I parse the .cs dump files
3. Map old → new addresses
4. Update your source file
5. Send back results with comments

⚡ *Quick Start:*
Click 'Upload Files' or just send me your files!
"""
    
    await update.message.reply_text(
        welcome_text,
        parse_mode='Markdown',
        reply_markup=get_main_keyboard()
    )
    logger.info(f"User {user_id} started bot")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    help_text = """
📚 *Help & Instructions*

*Supported File Types:*
• Source: `.cpp`, `.c`, `.h`, `.hpp`, `.cc`, `.cxx`, `.txt`
• Dumps: `.cs`, `.txt`, `.log`, `.dump`

*Required .cs Dump Format:*
Your .cs files should contain function addresses like:


*Limits:*
• Max file size: 50MB
• Supported hex offsets: 0x000000 to 0xFFFFFF...

*Commands:*
/start - Start the bot
/help - Show this help
/status - Check current status
/upload - Upload files
/clear - Clear all files

*Troubleshooting:*
• Ensure .cs files have RVA addresses
• Check file extensions
• Files must be under 50MB
"""
    
    if update.message:
        await update.message.reply_text(help_text, parse_mode='Markdown')
    elif update.callback_query:
        await update.callback_query.message.reply_text(help_text, parse_mode='Markdown')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command"""
    user_id = update.effective_user.id
    
    if user_id not in ALLOWED_USER_IDS:
        return
    
    session = user_sessions.get(user_id)
    
    if not session:
        status_text = "❌ No active session. Use /start to begin."
    else:
        status_text = f"""
📊 *Current Status*

{session.get_status_text()}

🔄 *State:* {session.status}
⏰ *Last activity:* {session.last_activity.strftime('%H:%M:%S')}

{'✅ Ready to process!' if session.is_ready() else '❌ Missing files'}
"""
    
    await update.message.reply_text(
        status_text,
        parse_mode='Markdown',
        reply_markup=get_main_keyboard()
    )

# ============ FILE HANDLING ============
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming document uploads
    THIS IS WHERE THE BOT READS YOUR FILES
    """
    user_id = update.effective_user.id
    
    if user_id not in ALLOWED_USER_IDS:
        return
    
    # Get session
    session = user_sessions.get(user_id)
    if not session:
        session = UserSession(user_id)
        user_sessions[user_id] = session
    
    # Get file info
    document = update.message.document
    file_name = document.file_name or "unknown"
    file_size = document.file_size or 0
    
    logger.info(f"User {user_id} uploading: {file_name} ({file_size} bytes)")
    
    # Check file size
    if file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ File too large: {file_size:,} bytes\n"
            f"Max size: {MAX_FILE_SIZE:,} bytes"
        )
        return
    
    # Download the file
    try:
        file = await context.bot.get_file(document.file_id)
        upload_dir = session.temp_dir / "uploads"
        file_path = upload_dir / file_name
        
        await file.download_to_drive(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Determine file type based on extension and content
        file_type = await determine_file_type(file_path, file_name)
        
        # Store based on type
        if file_type == "source":
            if session.source_file and session.source_file.exists():
                session.source_file.unlink()  # Delete old source file
            session.source_file = file_path
            action = "source file"
            
        elif file_type == "old_dump":
            if session.old_dump and session.old_dump.exists():
                session.old_dump.unlink()
            session.old_dump = file_path
            action = "OLD dump"
            
        elif file_type == "new_dump":
            if session.new_dump and session.new_dump.exists():
                session.new_dump.unlink()
            session.new_dump = file_path
            action = "NEW dump"
            
        else:
            action = "file (type unknown)"
        
        # Update session
        session.last_activity = datetime.now()
        session.status = f"received_{file_type}"
        
        # Send confirmation
        confirm_text = f"""
✅ *File Uploaded Successfully!*

📁 *File:* {file_name}
📏 *Size:* {file_size:,} bytes
📋 *Type:* {action}
🕒 *Time:* {datetime.now().strftime('%H:%M:%S')}

{session.get_status_text()}

{'🎯 *All files ready!* Click Start Update below.' if session.is_ready() else '📤 Send more files...'}
"""
        
        await update.message.reply_text(
            confirm_text,
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )
        
    except Exception as e:
        logger.error(f"Error handling file: {e}")
        await update.message.reply_text(
            f"❌ Error uploading file: {str(e)[:100]}"
        )

async def determine_file_type(file_path: Path, file_name: str) -> str:
    """
    Determine if file is source, old dump, or new dump
    YES - THIS READS YOUR FILE CONTENT TO DETERMINE TYPE!
    """
    file_extension = Path(file_name).suffix.lower()
    
    # Check by extension first
    if file_extension in ALLOWED_SOURCE_EXTENSIONS:
        return "source"
    
    elif file_extension in ALLOWED_DUMP_EXTENSIONS:
        # Read file content to determine old/new
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5KB
            
            # Check for indicators in filename
            file_lower = file_name.lower()
            if 'old' in file_lower or 'previous' in file_lower or 'v1' in file_lower:
                return "old_dump"
            elif 'new' in file_lower or 'updated' in file_lower or 'v2' in file_lower:
                return "new_dump"
            
            # Check content for dump patterns
            dump_patterns = [
                r'RVA:\s*0x[0-9A-Fa-f]+',
                r'Offset:\s*0x[0-9A-Fa-f]+',
                r'//.*Function',
                r'public.*0x[0-9A-Fa-f]+'
            ]
            
            for pattern in dump_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Default to old dump if ambiguous
                    return "old_dump"
            
        except Exception as e:
            logger.warning(f"Error reading file for type detection: {e}")
    
    # Default based on extension
    if file_extension == '.cs':
        return "old_dump"
    
    return "unknown"

# ============ .CS DUMP PARSING ============
def parse_cs_dump_file(dump_file: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse .cs dump file for function addresses
    RETURNS: (function_name -> address, address -> function_name)
    """
    func_to_addr = {}
    addr_to_func = {}
    
    try:
        with open(dump_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        logger.info(f"Parsing .cs dump: {dump_file.name} ({len(content)} chars)")
        
        # PATTERN 1: // FunctionName\n// RVA: 0x123456
        pattern1 = r'//\s*([A-Za-z_][A-Za-z0-9_]*)\s*\r?\n//\s*RVA:\s*(0x[0-9A-Fa-f]+)'
        matches1 = re.findall(pattern1, content, re.IGNORECASE | re.MULTILINE)
        
        for func_name, address in matches1:
            addr_lower = address.lower()
            func_to_addr[func_name] = addr_lower
            addr_to_func[addr_lower] = func_name
            logger.debug(f"Pattern1: {func_name} -> {addr_lower}")
        
        # PATTERN 2: FunctionName RVA: 0x123456
        pattern2 = r'([A-Za-z_][A-Za-z0-9_]*)\s+RVA:\s*(0x[0-9A-Fa-f]+)'
        matches2 = re.findall(pattern2, content, re.IGNORECASE)
        
        for func_name, address in matches2:
            addr_lower = address.lower()
            if func_name not in func_to_addr:
                func_to_addr[func_name] = addr_lower
                addr_to_func[addr_lower] = func_name
                logger.debug(f"Pattern2: {func_name} -> {addr_lower}")
        
        # PATTERN 3: // RVA: 0x123456 (FunctionName)
        pattern3 = r'//\s*RVA:\s*(0x[0-9A-Fa-f]+)\s*\(([A-Za-z_][A-Za-z0-9_]*)\)'
        matches3 = re.findall(pattern3, content, re.IGNORECASE)
        
        for address, func_name in matches3:
            addr_lower = address.lower()
            if func_name not in func_to_addr:
                func_to_addr[func_name] = addr_lower
                addr_to_func[addr_lower] = func_name
                logger.debug(f"Pattern3: {func_name} -> {addr_lower}")
        
        # PATTERN 4: Offset: 0x123456 (in C# dumps)
        pattern4 = r'([A-Za-z_][A-Za-z0-9_]*).*?Offset:\s*(0x[0-9A-Fa-f]+)'
        matches4 = re.findall(pattern4, content, re.IGNORECASE | re.DOTALL)
        
        for func_name, address in matches4:
            addr_lower = address.lower()
            if func_name not in func_to_addr and len(func_name) > 2:
                func_to_addr[func_name] = addr_lower
                addr_to_func[addr_lower] = func_name
                logger.debug(f"Pattern4: {func_name} -> {addr_lower}")
        
        # PATTERN 5: Static field offsets in C#
        pattern5 = r'public static.*?(0x[0-9A-Fa-f]+).*?//\s*([A-Za-z_][A-Za-z0-9_]*)'
        matches5 = re.findall(pattern5, content, re.IGNORECASE)
        
        for address, func_name in matches5:
            addr_lower = address.lower()
            if func_name not in func_to_addr:
                func_to_addr[func_name] = addr_lower
                addr_to_func[addr_lower] = func_name
                logger.debug(f"Pattern5: {func_name} -> {addr_lower}")
        
        logger.info(f"Parsed {len(func_to_addr)} functions from {dump_file.name}")
        
        if not func_to_addr:
            logger.warning(f"No functions found in {dump_file.name}")
            # Try raw hex search as fallback
            hex_pattern = r'(0x[0-9A-Fa-f]{6,})'
            hex_matches = re.findall(hex_pattern, content, re.IGNORECASE)
            for i, addr in enumerate(hex_matches[:100]):  # Limit to first 100
                func_name = f"Function_{i:03d}"
                addr_lower = addr.lower()
                func_to_addr[func_name] = addr_lower
                addr_to_func[addr_lower] = func_name
        
        return func_to_addr, addr_to_func
        
    except Exception as e:
        logger.error(f"Error parsing .cs dump {dump_file}: {e}")
        return {}, {}

# ============ OFFSET PROCESSING ============
def extract_offsets_from_source(source_file: Path) -> List[Dict]:
    """Extract all hex offsets from source file"""
    offsets = []
    
    try:
        with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Find all hex offsets (6+ hex digits)
            matches = re.finditer(r'(0x[0-9A-Fa-f]{6,})', line, re.IGNORECASE)
            
            for match in matches:
                offset = match.group(1)
                start_pos = match.start()
                
                # Skip if in string literal
                if '"' in line[:start_pos] and line[:start_pos].count('"') % 2 == 1:
                    continue
                
                # Skip if in single line comment
                if '//' in line[:start_pos]:
                    comment_pos = line.find('//')
                    if comment_pos < start_pos:
                        continue
                
                offsets.append({
                    'line': line_num,
                    'offset': offset,
                    'original_line': line.rstrip('\n'),
                    'position': start_pos
                })
        
        logger.info(f"Found {len(offsets)} offsets in {source_file.name}")
        return offsets
        
    except Exception as e:
        logger.error(f"Error extracting offsets: {e}")
        return []

def create_offset_mapping(
    offsets: List[Dict],
    old_addr_to_func: Dict[str, str],
    new_func_to_addr: Dict[str, str]
) -> Dict[str, Dict]:
    """Create mapping from old offsets to new offsets"""
    mapping = {}
    
    unique_offsets = set(offset['offset'] for offset in offsets)
    
    for offset in unique_offsets:
        offset_lower = offset.lower()
        
        # Find function for this offset
        func_name = old_addr_to_func.get(offset_lower)
        
        if func_name:
            # Get new offset for this function
            new_offset = new_func_to_addr.get(func_name)
            
            if new_offset:
                # Preserve original case
                if offset.isupper():
                    new_offset = new_offset.upper()
                elif offset[2:].isupper():  # If hex part is uppercase
                    new_offset = new_offset.upper()
                
                mapping[offset] = {
                    'new_offset': new_offset,
                    'function': func_name,
                    'changed': offset_lower != new_offset.lower()
                }
                logger.debug(f"Mapped: {offset} -> {new_offset} ({func_name})")
            else:
                logger.debug(f"Function {func_name} not found in new dump")
        else:
            logger.debug(f"No function found for offset {offset}")
    
    logger.info(f"Created mapping for {len(mapping)} offsets")
    return mapping

def update_source_file(
    source_file: Path,
    mapping: Dict[str, Dict],
    offsets: List[Dict],
    output_dir: Path
) -> Dict:
    """Update source file with new offsets and comments"""
    try:
        with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        changes = []
        updated_lines = set()
        
        for offset_info in offsets:
            line_num = offset_info['line'] - 1
            old_offset = offset_info['offset']
            
            if old_offset in mapping:
                info = mapping[old_offset]
                
                if info['changed'] and line_num not in updated_lines:
                    line = lines[line_num]
                    original_line = line.rstrip('\n')
                    
                    # Replace offset (case-sensitive replacement)
                    offset_pattern = re.compile(re.escape(old_offset), re.IGNORECASE)
                    new_line = offset_pattern.sub(info['new_offset'], line)
                    new_line = new_line.rstrip('\n')
                    
                    # Add comment if not already there
                    if '//' not in original_line or original_line.find('//') > offset_info['position']:
                        comment = f"  // {old_offset} -> {info['new_offset']} ({info['function']})"
                        new_line += comment
                    
                    lines[line_num] = new_line + '\n'
                    updated_lines.add(line_num)
                    
                    changes.append({
                        'line': line_num + 1,
                        'old': old_offset,
                        'new': info['new_offset'],
                        'function': info['function']
                    })
        
        # Save updated file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        updated_filename = f"{source_file.stem}_UPDATED_{timestamp}{source_file.suffix}"
        updated_file = output_dir / updated_filename
        
        with open(updated_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        # Create change log
        log_file = create_change_log(changes, mapping, updated_file, output_dir)
        
        # Create summary
        summary_file = create_summary_file(changes, mapping, source_file, updated_file, output_dir)
        
        return {
            'success': True,
            'updated_file': updated_file,
            'log_file': log_file,
            'summary_file': summary_file,
            'changes': changes,
            'total_changed': len(changes),
            'total_offsets': len(set(offset['offset'] for offset in offsets))
        }
        
    except Exception as e:
        logger.error(f"Error updating source file: {e}")
        return {'success': False, 'error': str(e)}

def create_change_log(changes: List[Dict], mapping: Dict, updated_file: Path, output_dir: Path) -> Path:
    """Create detailed change log"""
    log_file = output_dir / f"{updated_file.stem}_CHANGES.log"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("OFFSET UPDATE CHANGE LOG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Updated file: {updated_file.name}\n")
        f.write(f"Total changes: {len(changes)}\n\n")
        
        # Group changes by function
        by_function = defaultdict(list)
        for change in changes:
            by_function[change['function']].append(change)
        
        f.write("CHANGES BY FUNCTION:\n")
        f.write("=" * 80 + "\n")
        
        for func_name, func_changes in sorted(by_function.items()):
            f.write(f"\nFunction: {func_name}\n")
            f.write("-" * 40 + "\n")
            for change in func_changes:
                f.write(f"Line {change['line']:4d}: {change['old']} → {change['new']}\n")
        
        f.write("\n\nALL CHANGES (chronological):\n")
        f.write("=" * 80 + "\n")
        for change in changes:
            f.write(f"Line {change['line']:4d}: {change['old']} → {change['new']} ({change['function']})\n")
    
    return log_file

def create_summary_file(changes: List[Dict], mapping: Dict, source_file: Path, updated_file: Path, output_dir: Path) -> Path:
    """Create summary file"""
    summary_file = output_dir / f"{updated_file.stem}_SUMMARY.txt"
    
    changed_offsets = sum(1 for info in mapping.values() if info['changed'])
    unchanged_offsets = len(mapping) - changed_offsets
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("OFFSET UPDATE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Source file: {source_file.name}\n")
        f.write(f"Updated file: {updated_file.name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("STATISTICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total offsets found: {len(set(mapping.keys()))}\n")
        f.write(f"Offsets changed: {changed_offsets}\n")
        f.write(f"Lines modified: {len(changes)}\n")
        f.write(f"Functions affected: {len(set(change['function'] for change in changes))}\n")
        f.write(f"Unchanged offsets: {unchanged_offsets}\n\n")
        
        f.write("CHANGED OFFSETS:\n")
        f.write("-" * 50 + "\n")
        for old_offset, info in mapping.items():
            if info['changed']:
                f.write(f"{old_offset} → {info['new_offset']} ({info['function']})\n")
    
    return summary_file

# ============ PROCESSING HANDLER ============
async def process_files_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle processing request"""
    user_id = update.effective_user.id
    
    if user_id not in ALLOWED_USER_IDS:
        return
    
    session = user_sessions.get(user_id)
    
    if not session:
        await update.message.reply_text("❌ No active session. Use /start first.")
        return
    
    if not session.is_ready():
        await update.message.reply_text(
            "❌ Missing files!\n\n" + session.get_status_text(),
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )
        return
    
    # Start processing
    await update.message.reply_text(
        "🔄 *Starting offset update...*\n\n"
        "1. Parsing .cs dump files...\n"
        "2. Extracting offsets...\n"
        "3. Creating mapping...\n"
        "4. Updating source file...\n\n"
        "⏳ This may take a few seconds...",
        parse_mode='Markdown',
        reply_markup=get_processing_keyboard()
    )
    
    # Run processing in background
    asyncio.create_task(process_files_background(session, update, context))

async def process_files_background(session: UserSession, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Background file processing"""
    try:
        session.status = "processing"
        
        # Parse dump files
        old_func_to_addr, old_addr_to_func = parse_cs_dump_file(session.old_dump)
        new_func_to_addr, _ = parse_cs_dump_file(session.new_dump)
        
        if not old_func_to_addr or not new_func_to_addr:
            await update.message.reply_text(
                "❌ Could not parse .cs dump files.\n"
                "Make sure they contain RVA addresses in the correct format.",
                parse_mode='Markdown'
            )
            return
        
        # Extract offsets
        offsets = extract_offsets_from_source(session.source_file)
        
        if not offsets:
            await update.message.reply_text(
                "❌ No hex offsets found in source file.\n"
                "Make sure your source file contains 0x123456 style offsets.",
                parse_mode='Markdown'
            )
            return
        
        # Create mapping
        mapping = create_offset_mapping(offsets, old_addr_to_func, new_func_to_addr)
        
        if not mapping:
            await update.message.reply_text(
                "❌ Could not map any offsets.\n"
                "Check if offsets in source match addresses in dump files.",
                parse_mode='Markdown'
            )
            return
        
        # Update source file
        output_dir = session.temp_dir / "outputs"
        result = update_source_file(session.source_file, mapping, offsets, output_dir)
        
        if not result['success']:
            await update.message.reply_text(
                f"❌ Error updating file: {result.get('error', 'Unknown error')}",
                parse_mode='Markdown'
            )
            return
        
        # Send results
        await send_results(session, update, context, result, mapping)
        
        session.status = "completed"
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        await update.message.reply_text(
            f"❌ Processing error: {str(e)[:200]}",
            parse_mode='Markdown'
        )
        session.status = "error"

async def send_results(
    session: UserSession,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    result: Dict,
    mapping: Dict
) -> None:
    """Send processing results to user"""
    changed_offsets = sum(1 for info in mapping.values() if info['changed'])
    total_offsets = result['total_offsets']
    
    # Summary message
    summary = f"""
✅ *Offset Update Complete!*

📊 *Statistics:*
• Total offsets found: {total_offsets}
• Offsets changed: {changed_offsets}
• Lines modified: {result['total_changed']}
• Functions updated: {len(set(info['function'] for info in mapping.values() if info['changed']))}
• Unchanged offsets: {total_offsets - changed_offsets}

📁 *Generated files:*
• Updated source file
• Detailed change log
• Summary report

📤 *Sending files...*
"""
    
    if hasattr(update, 'callback_query') and update.callback_query:
        await update.callback_query.message.edit_text(summary, parse_mode='Markdown')
    elif update.message:
        await update.message.edit_text(summary, parse_mode='Markdown')
    
    # Send files
    files_to_send = [
        (result['updated_file'], "📄 Updated source file"),
        (result['log_file'], "📋 Detailed change log"),
        (result['summary_file'], "📊 Summary report")
    ]
    
    for file_path, caption in files_to_send:
        try:
            with open(file_path, 'rb') as f:
                await context.bot.send_document(
                    chat_id=session.user_id,
                    document=f,
                    filename=file_path.name,
                    caption=caption
                )
            await asyncio.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Error sending file {file_path}: {e}")
    
    # Final message
    final_msg = """
🎉 *Processing Finished!*

All files have been sent successfully.

*What's next:*
1. Review the updated file
2. Check the change log for details
3. Test your updated code

*Need to process more files?*
Use /start to begin a new session!
"""
    
    await context.bot.send_message(
        chat_id=session.user_id,
        text=final_msg,
        parse_mode='Markdown',
        reply_markup=get_main_keyboard()
    )

# ============ BUTTON HANDLERS ============
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if user_id not in ALLOWED_USER_IDS:
        return
    
    action = query.data
    
    if action == "upload":
        await query.edit_message_text(
            "📤 *Upload your files*\n\n"
            "Please send me these files:\n"
            "1. 📄 C++ source file (.cpp/.c/.h)\n"
            "2. 📁 OLD .cs dump file\n"
            "3. 📁 NEW .cs dump file\n\n"
            "You can send them in any order.\n"
            "I'll automatically detect file types.",
            parse_mode='Markdown',
            reply_markup=get_file_type_keyboard()
        )
    
    elif action == "status":
        session = user_sessions.get(user_id)
        if session:
            await query.edit_message_text(
                session.get_status_text(),
                parse_mode='Markdown',
                reply_markup=get_main_keyboard()
            )
    
    elif action == "type_source":
        await query.edit_message_text(
            "📄 *Upload C++ Source File*\n\n"
            "Please send your C++ source file.\n"
            "Supported: .cpp, .c, .h, .hpp, .cc, .cxx\n\n"
            "The file should contain hex offsets like:\n"
            "`void* offset = 0x123456;`",
            parse_mode='Markdown'
        )
    
    elif action == "type_old":
        await query.edit_message_text(
            "📁 *Upload OLD .cs Dump File*\n\n"
            "Please send the OLD .cs dump file.\n"
            "It should contain function addresses like:\n"
            "```\n// FunctionName\n// RVA: 0x123456\n```",
            parse_mode='Markdown'
        )
    
    elif action == "type_new":
        await query.edit_message_text(
            "📁 *Upload NEW .cs Dump File*\n\n"
            "Please send the NEW .cs dump file.\n"
            "It should contain updated function addresses.",
            parse_mode='Markdown'
        )
    
    elif action == "back":
        await query.edit_message_text(
            "🤖 *Offset Updater Bot*\n\n"
            "Main menu:",
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )
    
    elif action == "clear":
        if user_id in user_sessions:
            user_sessions[user_id].cleanup()
            del user_sessions[user_id]
        await query.edit_message_text(
            "🗑️ *All files cleared!*\n\n"
            "Session reset. Use /start to begin again.",
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )

# ============ ERROR HANDLING ============
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")
    
    if update and update.effective_user:
        try:
            await context.bot.send_message(
                chat_id=update.effective_user.id,
                text="❌ An error occurred. Please try again or use /start."
            )
        except:
            pass

# ============ CLEANUP TASK ============
async def cleanup_old_sessions():
    """Clean up old user sessions periodically"""
    while True:
        try:
            now = datetime.now()
            to_remove = []
            
            for user_id, session in list(user_sessions.items()):
                # Remove sessions older than 1 hour
                if (now - session.last_activity).total_seconds() > 3600:
                    session.cleanup()
                    to_remove.append(user_id)
            
            for user_id in to_remove:
                del user_sessions[user_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old sessions")
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        
        await asyncio.sleep(300)  # Run every 5 minutes

# ============ MAIN ============
def main() -> None:
    """Start the bot"""
    # Check token
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ ERROR: Please set your Telegram Bot Token!")
        print("1. Create bot with @BotFather")
        print("2. Get token")
        print("3. Set BOT_TOKEN in code or environment variable")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("upload", start_command))
    application.add_handler(CommandHandler("process", process_files_command))
    
    # Add message handler for documents
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    # Add button handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Create temp directory
    Path("temp").mkdir(exist_ok=True)
    
    # Start cleanup task
    asyncio.get_event_loop().create_task(cleanup_old_sessions())
    
    # Start the bot
    print("🤖 Telegram Offset Updater Bot")
    print("=" * 40)
    print(f"Token: {BOT_TOKEN[:10]}...")
    print(f"Allowed users: {ALLOWED_USER_IDS}")
    print("Starting bot...")
    print("Press Ctrl+C to stop")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()