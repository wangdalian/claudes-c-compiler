//! ARMv7 peephole optimizer.
//! Performs simple pattern-matching optimizations on the generated assembly.

pub fn peephole_optimize(input: String) -> String {
    let mut lines: Vec<String> = input.lines().map(|l| l.to_string()).collect();
    let mut changed = true;

    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < lines.len() {
            // Remove redundant mov r0, r0
            if lines[i].trim() == "mov r0, r0" {
                lines.remove(i);
                changed = true;
                continue;
            }
            // Remove str rX, [slot] followed by ldr rX, [slot] (same slot)
            if let (Some(store_slot), Some(load_slot)) = (
                extract_str_slot(&lines[i]),
                extract_ldr_slot(&lines[i + 1]),
            ) {
                if store_slot == load_slot {
                    let store_reg = extract_str_reg(&lines[i]);
                    let load_reg = extract_ldr_reg(&lines[i + 1]);
                    if let (Some(sr), Some(lr)) = (store_reg, load_reg) {
                        if sr == lr {
                            lines.remove(i + 1);
                            changed = true;
                            continue;
                        }
                    }
                }
            }
            i += 1;
        }
    }

    lines.join("\n")
}

fn extract_str_slot(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("str ") {
        if let Some(pos) = rest.find('[') {
            return Some(&rest[pos..]);
        }
    }
    None
}

fn extract_ldr_slot(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("ldr ") {
        if let Some(pos) = rest.find('[') {
            return Some(&rest[pos..]);
        }
    }
    None
}

fn extract_str_reg(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("str ") {
        if let Some(pos) = rest.find(',') {
            return Some(rest[..pos].trim());
        }
    }
    None
}

fn extract_ldr_reg(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix("ldr ") {
        if let Some(pos) = rest.find(',') {
            return Some(rest[..pos].trim());
        }
    }
    None
}
