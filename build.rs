use std::io::Write;

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
    // get body
    let uri = "https://raw.githubusercontent.com/MingweiSamuel/Riven/v/2.x.x/riven/src/consts/champion.rs";
    println!("Getting {uri}...");
    let body = reqwest::blocking::get(uri)
        .unwrap()
        .text()
        .unwrap();
    println!("Response received.");
    
    // extract only the "enum" definition
    let regex = regex::Regex::new(r"(pub newtype_enum Champion\(i16\) \{\n)([^}]*)").unwrap();
    let captures = regex.captures(&body).unwrap();
    let body = &captures[2];
    println!("Body extracted: {body}");
    
    // buffer file
    let mut fd = std::io::BufWriter::new(fd);
    
    println!("Starting writes...");
    // write import
    fd.write_all("use riven::consts::Champion;\n".as_bytes()).unwrap();
    // write function head
    fd.write_all("/// This is sad.\npub fn champ_to_u8(champ: Champion) -> u8 {\n".as_bytes()).unwrap();
    // write body
    fd.write_all("\tmatch champ {\n".as_bytes()).unwrap();
    
    // write the champion map lines
    let mut i: isize = -1;
    for line in body.lines() {
        // prepare line
        let line = line.trim();
        if line.starts_with("///") || line.is_empty() {
            continue;
        }
        // skip the "NONE = -1" (first) entry
        else if i == -1 {
            i = 0;
            continue;
        }
        
        let mut builder = String::new();
        builder.push_str("\t\tChampion::");
        builder.push_str(line.split('=').next().unwrap());
        builder.push_str("=> ");
        builder.push_str(&i.to_string());
        builder.push(','); builder.push('\n');
        
        i += 1;
        fd.write_all(builder.as_bytes()).unwrap();
    }
    
    // add else clause
    fd.write_all("\t\tchamp => panic!(\"Champion {champ} not mapped!\")\n".as_bytes()).unwrap();
    
    // close function
    fd.write_all("\t}\n}".as_bytes()).unwrap();
    
    println!("Finished.")
}
