def generate_sinhala_dictionary(output_path="PaddleOCR/ppocr/utils/dict/sinhala_dict.txt"):
    """Generate complete Sinhala character set with proper Unicode ordering"""
    
    characters = []
    
    # Independent Vowels (U+0D85 to U+0D96)
    vowels = [
        'අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ',
        'ඍ', 'ඎ', 'ඏ', 'ඐ', 'එ', 'ඒ', 'ඓ', 'ඔ', 'ඕ', 'ඖ'
    ]
    
    # Consonants (U+0D9A to U+0DC6)
    consonants = [
        'ක', 'ខ', 'ග', 'ඝ', 'ඞ', 'ඟ',
        'ච', 'ඡ', 'ජ', 'ඣ', 'ඤ', 'ඥ', 'ඦ',
        'ට', 'ඨ', 'ඩ', 'ඪ', 'ණ', 'ඬ',
        'ත', 'ථ', 'ද', 'ධ', 'න', 'ඳ',
        'ප', 'ඵ', 'බ', 'භ', 'ම', 'ඹ',
        'ය', 'ර', 'ල', 'ව',
        'ශ', 'ෂ', 'ස', 'හ', 'ළ', 'ෆ'
    ]
    
    # Signs (U+0D82, U+0D83, U+0DCA)
    signs = ['ං', 'ඃ', '්']
    
    # Dependent Vowel Signs (U+0DCF to U+0DDF)
    dependent_vowels = [
        'ා', 'ැ', 'ෑ', 'ි', 'ී', 'ු', 'ූ', 'ෘ', 'ෲ',
        'ෟ', 'ෳ', 'ෙ', 'ේ', 'ෛ', 'ො', 'ෝ', 'ෞ', 'ෟ', 'ෲ', 'ෳ'
    ]
    
    # Zero-Width Characters (U+200C, U+200D)
    zero_width = ['\u200d', '\u200c']  # ZWJ, ZWNJ
    
    # Numbers
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Common punctuation
    punctuation = [' ', '.', ',', '?', '!', '-', ':', ';', '(', ')', '"', "'", '/']
    
    # Combine all
    characters.extend(vowels)
    characters.extend(consonants)
    characters.extend(signs)
    characters.extend(dependent_vowels)
    characters.extend(zero_width)
    characters.extend(numbers)
    characters.extend(punctuation)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chars = []
    for char in characters:
        if char not in seen:
            seen.add(char)
            unique_chars.append(char)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in unique_chars:
            f.write(char + '\n')
    
    print(f"✓ Generated Sinhala dictionary with {len(unique_chars)} characters")
    print(f"✓ Saved to: {output_path}")
    
    return unique_chars

if __name__ == "__main__":
    generate_sinhala_dictionary()