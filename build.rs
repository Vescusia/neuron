use std::io::Write;
use riven::consts::Champion;

fn main() {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("champ_to_u8.rs");
    println!("Opening file {dest_path:?}...");
    let fd = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(dest_path).unwrap();
    write_conversion(fd);

    println!("cargo::rerun-if-changed=Cargo.lock")
}

fn write_conversion(fd: std::fs::File)  {
    // buffer file
    let mut fd = std::io::BufWriter::new(fd);

    println!("Starting writes...");
    // write import
    fd.write_all("use riven::consts::Champion;\n".as_bytes()).unwrap();
    // write function head
    fd.write_all("/// This is sad.\npub fn champ_to_u8(champ: Champion) -> u8 {\n".as_bytes()).unwrap();
    // write body
    fd.write_all("\tmatch champ {\n".as_bytes()).unwrap();

    // sort champs by order of addition (somewhat)
    let mut all_known = Champion::ALL_KNOWN;
    all_known.sort_unstable();

    // write the champion map lines
    let mut i: u8 = 0;
    for champ in all_known {
        if champ == Champion::NONE {
            continue
        }

        // build line
        let mut builder = String::new();
        builder.push_str("\t\tChampion::");

        // sanitize name
        let name = match champ {
            Champion::LE_BLANC => "le_blanc",
            c => c.name().unwrap()
        }.to_uppercase()
            .replace(['\'', ' ', '&', '.'], "_");
        // remove multiple '_'
        let chars: Vec<char> = name.chars().collect();
        let mut name: String = chars.windows(2)
            .filter(|s| s[0] != '_' || s[1] != '_')
            .map(|s| s[0])
            .collect();
        name.push(*chars.last().unwrap());
        builder.push_str(&name);
        
        // map to u8
        builder.push_str(" => ");
        builder.push_str(&i.to_string());

        // finish line
        builder.push(','); builder.push('\n');
        fd.write_all(builder.as_bytes()).unwrap();
        i += 1;
    }

    // add else clause
    fd.write_all("\t\tchamp => panic!(\"Champion {champ} not mapped!\")\n".as_bytes()).unwrap();
    
    // close function
    fd.write_all("\t}\n}".as_bytes()).unwrap();
    
    println!("Finished.")
}
