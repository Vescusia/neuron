use std::io::{Read, Seek, Write};
use riven::consts::Champion;

fn main() {
    let out_dir = std::env::var_os("OUT_DIR").unwrap();
    let ctu_dest_path = std::path::Path::new(&out_dir).join("champ_to_u8.rs");
    let utc_dest_path = std::path::Path::new(&out_dir).join("u8_to_champ.rs");
    println!("Opening file {ctu_dest_path:?}...");
    let ctu_fd = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(ctu_dest_path).unwrap();
    println!("Opening file {utc_dest_path:?}...");
    let utc_fd = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(utc_dest_path).unwrap();
    write_conversion(ctu_fd, utc_fd);

    println!("cargo::rerun-if-changed=Cargo.lock")
}

fn write_conversion(mut ctu_fd: std::fs::File, mut utc_fd: std::fs::File)  {
    // converting champions to u8's
    {
        // buffer file
        let mut fd = std::io::BufWriter::new(&mut ctu_fd);

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
            builder.push(',');
            builder.push('\n');
            fd.write_all(builder.as_bytes()).unwrap();
            i += 1;
        }

        // add else clause
        fd.write_all("\t\tchamp => panic!(\"Champion {champ} not mapped!\")\n".as_bytes()).unwrap();

        // close function
        fd.write_all("\t}\n}".as_bytes()).unwrap();
    }

    // also creating the inverse file
    {
        // read from ctu file
        let mut buf = String::with_capacity(1 << 14);
        ctu_fd.seek(std::io::SeekFrom::Start(0)).unwrap();
        ctu_fd.read_to_string(&mut buf).expect("Invalid utf-8 in champ_to_u8.rs");
        
        // replace func definition
        let buf = buf.replace("champ_to_u8(champ: Champion) -> u8", "u8_to_champion(champ: u8) -> Champion");
        let buf = buf.replace("use riven::consts::Champion;\n", "");
        
        // swap champ => u8 to u8 => champ
        let re = regex::Regex::new(r"(\S+) => ([0-9]+),").unwrap();
        let new = re.replace_all(&buf, "$2 => $1,");
        
        // write
        utc_fd.write_all(new.as_bytes()).unwrap();
    }

    println!("Finished.")
}
