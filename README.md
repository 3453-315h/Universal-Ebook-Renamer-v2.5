📚 Universal Ebook Renamer v2.5 (PDF, EPUB, MOBI, FB2, AZW)

Intelligent batch renaming tool for eBook collections with metadata extraction and Google Books API integration

    🎯 Features Overview
      📖 Supported Formats
        📄 PDF - Portable Document Format
        📘 EPUB - Electronic Publication
        📱 MOBI - Mobipocket eBook
        🔥 AZW/AZW3 - Amazon Kindle formats
        📚 FB2 - FictionBook XML format
      🔧 Smart Processing Engine
        🤖 OCR Error Correction - Automatically fixes common OCR mistakes (O→0, I→1, S→5, etc.)
        🔍 Google Books API Integration - Enriches metadata via online cross-referencing
        🌐 Online Data Preference - Optionally prioritize API results over local metadata
        💾 Automatic Backups - Creates backups before any renaming operation
        🚨 Intelligent Duplicate Detection - Interactive conflict resolution with multiple options
      🏷️ Advanced Filename Formatting
        🏷️ Custom Templates - Flexible placeholders: {title}, {author}, {year}, {isbn}
        🎨 Case Transformation - Choose between original, lower, upper, or title case
        ⬜ Space Replacement - Convert spaces to underscores, dashes, or dots
        📏 Smart Length Limiting - Gracefully truncates long filenames
        🔗 "And" Replacement - Replace "and" with "&" in titles (e.g., "War and Peace" → "War & Peace")
      🎮 User Experience
        🎮 Interactive Mode - Prompt for duplicate handling (Skip/Rename/All/Quit)
        📜 File Logging - Save complete operation logs to disk
        🎨 Rich Terminal Output - Colored status indicators and progress tracking
        🔊 Verbose/quiet modes - Control output detail level





⚡ Quick Start

    Installation
    # Clone or download the script
    wget https://example.com/UERv2.5.py
    chmod +x UERv2.5.py
    # The script auto-installs required dependencies (requests, PyPDF2)

    Basic Usage
    # Rename all eBooks in current directory using title only
    python UERv2.5.py

    # Process a specific directory
    python UERv2.5.py "~/My Books"

    # Preview changes without modifying files
    python UERv2.5.py "~/Books" --dry-run



⚙️ Complete CLI Reference

📄 Format Selection Switches

    Table
    Switch   Icon	Description	              Example
    --pdf	  📄	Include PDF files only	python UERv2.5.py --pdf
    --no-pdf  🚫	Exclude PDF files	    python UERv2.5.py --no-pdf
    --epub	  📘	Include EPUB files      python UERv2.5.py --epub
    --no-epub 🚫	Exclude EPUB files	    python UERv2.5.py --no-epub
    --mobi	  📱	Include MOBI files	    python UERv2.5.py --mobi
    --no-mobi 🚫	Exclude MOBI files	    python UERv2.5.py --no-mobi
    --azw	  🔥	Include AZW files	    python UERv2.5.py --azw
    --no-azw  🚫	Exclude AZW files	    python UERv2.5.py --no-azw
    --azw3	  🔥	Include AZW3 files	    python UERv2.5.py --azw3
    --no-azw3 🚫	Exclude AZW3 files	    python UERv2.5.py --no-azw3
    --fb2	  📚	Include FB2 files	    python UERv2.5.py --fb2
    --no-fb2  🚫	Exclude FB2 files	    python UERv2.5.py --no-fb2


Combined Examples:

    # Process only PDF and EPUB
    python UERv2.5.py "~/Books" --pdf --epub

    # Process all formats except MOBI
    python UERv2.5.py "~/Books" --no-mobi

    # Process only AZW3 files with specific template
    python UERv2.5.py "./kindle" --azw3 --template "[{isbn}] {title}"


🏷️ Template & Naming Switches

    Table
    Switch	                       Icon	  Description                                      	Example
    --template	                    🏷️	Filename template with placeholders            	--template "{title} - {author} ({year})"
    --case	                        🎨	Transform case: original, lower, upper, title	--case title
    --max-length	                📏	Maximum filename length (default: 100)	        --max-length 80
    --replace-spaces	            ⬜	Replace spaces: none, underscore, dash, dot	    --replace-spaces dash
    --replace-and-with-ampersand	🔗	Replace "and" with "&" in titles	            --replace-and-with-ampersand




Template Placeholders:
     
    {title} - Book title
    {author} - Author name(s)
    {year} - Publication year
    {isbn} - ISBN number


Template Examples:

    # Simple title only
    --template "{title}"

    # Title and author
    --template "{title} - {author}"

    # Author first, then title
    --template "{author} - {title}"

    # Include year in parentheses
    --template "{title} ({year})"

    # ISBN prefix format
    --template "[{isbn}] {title}"

    # Complex directory structure
    --template "{author}/{title} [{year}]"


🔍 Data Processing Switches

    Table
    Switch	        Icon	Description	                    Example
    --no-ocr	    🤖	Disable OCR error correction	    --no-ocr
    --online-search	🔍	Enable Google Books API lookup	    --online-search
    --online-prefer	🌐	Prefer API data over local metadata	--online-prefer --online-search

Examples:

    # Use Google Books API for all files
    python UERv2.5.py "~/Books" --online-search

    # Prefer online data, but keep local as fallback
    python UERv2.5.py "~/Books" --online-search --online-prefer

    # Disable OCR (faster but less accurate for scanned PDFs)
    python UERv2.5.py "~/Books" --no-ocr


🎮 Operation Mode Switches

    Table
    Switch              	Icon	Description	                      Example
    -b, --no-backup	      💾    	Skip creating backups            	-b
    -n, --dry-run	      🔍	    Preview changes only	            -n
    -f, --force	          ⚡   	    Overwrite duplicates automatically	-f
    --non-interactive  	  🎮	    Auto-skip all duplicates	        --non-interactive


Examples:

    # Preview changes without modifying files
    python UERv2.5.py "~/Books" -n --template "{title} - {author}"

    # Force overwrite all duplicates
    python UERv2.5.py "~/Books" -f

    # Fast bulk processing (no backups, non-interactive)
    python UERv2.5.py "~/Books" -b --non-interactive

    # Safe mode with backups and dry run
    python UERv2.5.py "~/Books" --dry-run --verbose




📊 Output Control Switches

      Table
      Switch	Icon	Description	                  Example
    --verbose	🔊	Enable verbose output (default)	--verbose
    --quiet	    🔕	Disable verbose output	        --quiet
    --no-color	🎨	Disable colored output        	--no-color
    --log-file	📜	Save log to file	            --log-file rename.log


Examples:

    # Quiet mode with logging
    python UERv2.5.py "~/Books" --quiet --log-file "batch_rename.log"

    # No colors for log parsing
    python UERv2.5.py "~/Books" --no-color --log-file "clean.log"


🎯 Workflow Examples

Example 1: Academic Papers

    python UERv2.5.py "~/Papers" \
    --pdf \
    --template "{author} - {title} ({year})" \
    --case title \
    --max-length 120 \
    --online-search \
    --log-file "papers.log"


Example 2: Fiction Library

    python UERv2.5.py "~/Fiction" \
    --epub --mobi --fb2 \
    --template "{title} - {author}" \
    --replace-and-with-ampersand \
    --online-search \
    --online-prefer \
    --backup

Example 3: ISBN-based Archival

    python UERv2.5.py "~/Archive" \
    --pdf --epub \
    --template "[{isbn}] {title}" \
    --replace-spaces underscore \
    --max-length 80 \
    --no-color \
    --non-interactive \
    --log-file "archive.log"

Example 4: Quick & Dirty Bulk Rename

    python UERv2.5.py "./inbox" \
    --no-backup \
    --non-interactive \
    --quiet \
    --template "{title}" \
    --max-length 60


Example 5: Conservative Safe Mode
      
      python UERv2.5.py "~/Books" \
      --dry-run \
      --verbose \
      --template "{title} - {author}" \
      --case original \
      --log-file "safe_test.log"



🔍 Status Icon Legend

   During operation, you'll see:
   
    📄 Processing: filename - Current file being processed
    🔍 Year found on page X - Deep scan discovered metadata
    📅 2023 - Extracted publication year
    👤 John Doe - Extracted author name
    📖 9781234567890 - Extracted ISBN
    🔍 Found online: Title... - Google Books API match found
    🌐 Using online data exclusively - Online data override active
    ✅ Renamed to: New Name.pdf - Successful rename
    ⚠️ Could not extract title - Warning - using filename fallback
    ❌ Error: message - Processing error occurred
    🚨 DUPLICATE: 'Name.pdf' already exists! - Duplicate filename detected
    ⏭️ Skipping filename - File skipped by user or non-interactive mode
    ℹ️ Already correctly named - No change needed
    💾 Backups will be saved to: path - Backup directory location


📊 Final Summary Output

    After completion, you'll see:

    📊 Summary: 145 renamed, 3 errors, 12 skipped

    Shows the total count of successfully renamed files, errors encountered, and files skipped due to duplicates or user choice.


🔧 Requirements

    Python 3.6+

    Auto-installed dependencies: requests, PyPDF2

    Optional: Internet connection for Google Books API features


📄 License

    This tool is provided as-is for personal and commercial use. Modify and distribute freely.

    Happy Organizing! 📚✨
