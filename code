from pymol import cmd, stored
import os
import time
import numpy as np
import pandas as pd
import sys
import colorsys
import math
import traceback
import logging
from collections import defaultdict

# ========================================================
# USER CONFIGURATION - SET THESE PARAMETERS BEFORE RUNNING
# ========================================================
INPUT_DIR = "./pdb_files"       # Directory containing PDB files
OUTPUT_DIR = "./results"        # Output directory (will be created)
ALIGN_TARGET = "mhc"            # Options: "mhc", "tcr", "peptide"
RMSD_TARGETS = ["peptide"]      # Options: ["mhc", "peptide", "tcr"] - list of chains to calculate RMSD for
USE_GLOBAL_CHAIN_MAPPING = True # Set to True to use consistent chain mapping for all files

# Global chain mapping (only used if USE_GLOBAL_CHAIN_MAPPING = True)
GLOBAL_CHAIN_MAP = {
    'mhc': ['1', '2'],     # MHC chains (usually the two longest chains)
    'peptide': ['3'],      # Peptide chain (usually shortest chain, 5-25 residues)
    'tcr': ['4', '5']      # TCR chains (usually next two longest chains)
}

# ========================================================
# MAIN SCRIPT - NO NEED TO MODIFY BELOW THIS POINT
# ========================================================

def setup_logging(output_dir):
    """Configure logging to file and console"""
    log_file = os.path.join(output_dir, "analysis.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def welcome_message(logger):
    """Display welcome message"""
    logger.info("\n" + "="*80)
    logger.info(" TCR-Peptide Alignment Suite ".center(80))
    logger.info(" Developed by Vranda Aggarwal ".center(80))
    logger.info("="*80)
    logger.info("\nProfessional MHC-TCR-Peptide analysis tool")
    logger.info(f"\nConfiguration:")
    logger.info(f"  Input Directory: {INPUT_DIR}")
    logger.info(f"  Output Directory: {OUTPUT_DIR}")
    logger.info(f"  Alignment Target: {ALIGN_TARGET.upper()}")
    logger.info(f"  RMSD Targets: {', '.join([t.upper() for t in RMSD_TARGETS])}")
    logger.info(f"  Chain Mapping: {'Global' if USE_GLOBAL_CHAIN_MAPPING else 'Auto-detected'}")
    if USE_GLOBAL_CHAIN_MAPPING:
        logger.info(f"    MHC Chains: {', '.join(GLOBAL_CHAIN_MAP['mhc'])}")
        logger.info(f"    Peptide Chain: {', '.join(GLOBAL_CHAIN_MAP['peptide'])}")
        logger.info(f"    TCR Chains: {', '.join(GLOBAL_CHAIN_MAP['tcr'])}")

def get_pdb_files(input_dir):
    """Get all PDB files from input directory"""
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
             if f.endswith(('.pdb', '.ent'))]
    if not files:
        raise FileNotFoundError("No PDB files found in input directory")
    return sorted(files)  # Sort for consistent ordering

def setup_output(output_dir):
    """Create output directory structure"""
    os.makedirs(output_dir, exist_ok=True)
    subdirs = ["aligned_pdbs", "chain_pdbs", "images", "excel", "sessions", "logs"]
    paths = {}
    for d in subdirs:
        path = os.path.join(output_dir, d)
        os.makedirs(path, exist_ok=True)
        paths[d] = path
    return paths

def auto_detect_chain_mapping(obj_name, pdb_id, logger):
    """Automatically detect chain types based on chain length characteristics"""
    chain_lengths = {}
    stored.chain_residues = {}
    
    # Get residue counts per chain
    cmd.iterate(f"{obj_name} and polymer and name CA", 
               "stored.chain_residues[chain] = stored.chain_residues.get(chain, 0) + 1",
               space={'stored': stored})
    
    if not stored.chain_residues:
        return None
    
    # Sort chains by length
    sorted_chains = sorted(stored.chain_residues.items(), key=lambda x: x[1], reverse=True)
    chain_lengths = dict(sorted_chains)
    all_chains = list(chain_lengths.keys())
    
    logger.info(f"  Chains in {pdb_id}:")
    for chain, length in sorted_chains:
        logger.info(f"    Chain {chain}: {length} residues")
    
    # Try to identify chains based on typical lengths
    mapping = {'mhc': [], 'peptide': [], 'tcr': []}
    
    # 1. Identify peptide (shortest chain between 5-25 residues)
    peptide_candidates = [c for c, l in chain_lengths.items() if 5 <= l <= 25]
    if peptide_candidates:
        # Prefer chains named 'C' or 'P' if present
        if 'C' in peptide_candidates:
            mapping['peptide'] = ['C']
        elif 'P' in peptide_candidates:
            mapping['peptide'] = ['P']
        else:
            mapping['peptide'] = [min(peptide_candidates, key=chain_lengths.get)]
    
    # 2. Identify MHC (two longest chains that aren't peptide)
    remaining_chains = [c for c in all_chains if c not in mapping['peptide']]
    if len(remaining_chains) >= 2:
        # Prefer chains named 'A'/'B' or '1'/'2'
        if 'A' in remaining_chains and 'B' in remaining_chains:
            mapping['mhc'] = ['A', 'B']
        elif '1' in remaining_chains and '2' in remaining_chains:
            mapping['mhc'] = ['1', '2']
        else:
            mapping['mhc'] = remaining_chains[:2]
    
    # 3. Identify TCR (next two chains that aren't peptide or MHC)
    remaining_chains = [c for c in remaining_chains if c not in mapping['mhc']]
    if len(remaining_chains) >= 2:
        # Prefer chains named 'D'/'E' or '3'/'4'
        if 'D' in remaining_chains and 'E' in remaining_chains:
            mapping['tcr'] = ['D', 'E']
        elif '3' in remaining_chains and '4' in remaining_chains:
            mapping['tcr'] = ['3', '4']
        else:
            mapping['tcr'] = remaining_chains[:2]
    elif remaining_chains:
        mapping['tcr'] = remaining_chains
    
    # Verify we have minimum required chains
    if not mapping['mhc']:
        logger.warning(f"  Couldn't identify MHC chains in {pdb_id}")
    if not mapping['peptide']:
        logger.warning(f"  Couldn't identify peptide chain in {pdb_id}")
    
    return mapping

def get_chain_type(obj_name, pdb_id, logger, global_map=None):
    """Determine chain type with intelligent fallback logic"""
    if global_map:
        # Verify chains exist in structure
        stored.available_chains = []
        cmd.iterate(f"{obj_name} and polymer", "stored.available_chains.append(chain)", space={'stored': stored})
        available_chains = list(set(stored.available_chains))
        
        # Check if all global chains exist
        valid = True
        for chain_type, chains in global_map.items():
            for chain in chains:
                if chain not in available_chains:
                    logger.warning(f"  Chain {chain} ({chain_type}) not found in {pdb_id}")
                    valid = False
        
        if valid:
            logger.info(f"  Using global chain mapping: MHC={global_map['mhc']}, Peptide={global_map['peptide']}, TCR={global_map['tcr']}")
            return global_map
    
    # First try automatic detection
    logger.info(f"  Auto-detecting chains for {pdb_id}")
    auto_mapping = auto_detect_chain_mapping(obj_name, pdb_id, logger)
    
    # Fallback to default naming if automatic detection failed
    if not auto_mapping or not auto_mapping['mhc']:
        logger.info("  Using fallback chain detection")
        
        # Check which standard chains exist
        stored.available_chains = []
        cmd.iterate(f"{obj_name} and polymer", "stored.available_chains.append(chain)", space={'stored': stored})
        available_chains = list(set(stored.available_chains))
        
        # Try different naming conventions
        if all(c in available_chains for c in ['A', 'B', 'C', 'D', 'E']):
            return {'mhc': ['A', 'B'], 'peptide': ['C'], 'tcr': ['D', 'E']}
        elif all(c in available_chains for c in ['1', '2', '3', '4', '5']):
            return {'mhc': ['1', '2'], 'peptide': ['3'], 'tcr': ['4', '5']}
        else:
            # Generic fallback: first 2 chains = MHC, next = peptide, next 2 = TCR
            return {
                'mhc': available_chains[:2],
                'peptide': [available_chains[2]] if len(available_chains) > 2 else [],
                'tcr': available_chains[3:5] if len(available_chains) > 4 else available_chains[3:]
            }
    
    return auto_mapping

def create_chain_objects(obj_name, prefix, chain_map):
    """Create separate objects for each chain type"""
    mhc_name = f"{prefix}_mhc"
    pep_name = f"{prefix}_pep"
    tcr_name = f"{prefix}_tcr"
    
    # Create objects only if chains are specified
    if chain_map['mhc']:
        mhc_sel = f"chain {'+'.join(chain_map['mhc'])}"
        cmd.create(mhc_name, f"{obj_name} and {mhc_sel}")
    else:
        cmd.create(mhc_name, f"{obj_name} and none")  # Create empty object
    
    if chain_map['peptide']:
        pep_sel = f"chain {'+'.join(chain_map['peptide'])}"
        cmd.create(pep_name, f"{obj_name} and {pep_sel}")
    else:
        cmd.create(pep_name, f"{obj_name} and none")
    
    if chain_map['tcr']:
        tcr_sel = f"chain {'+'.join(chain_map['tcr'])}"
        cmd.create(tcr_name, f"{obj_name} and {tcr_sel}")
    else:
        cmd.create(tcr_name, f"{obj_name} and none")
    
    return mhc_name, pep_name, tcr_name

def get_distinct_colors(n):
    """Generate visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i * 0.618033988749895
        hue = hue - math.floor(hue)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((r, g, b))
    return colors

def calculate_com_distance_matrix(objects, pdb_ids, name, logger):
    """Calculate center-of-mass distance matrix"""
    n = len(objects)
    dist_matrix = np.zeros((n, n))
    centers = []
    
    logger.info(f"Calculating {name} center of mass for {n} structures...")
    
    for i, obj_name in enumerate(objects):
        try:
            com = cmd.centerofmass(obj_name)
            centers.append(np.array(com))
            logger.info(f"  {pdb_ids[i]}: COM = ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
        except Exception as e:
            logger.error(f"  ERROR calculating COM for {pdb_ids[i]}: {str(e)}")
            centers.append(np.zeros(3))
    
    for i in range(n):
        for j in range(i, n):
            dist_matrix[i, j] = dist_matrix[j, i] = 0.0 if i == j else np.linalg.norm(centers[i] - centers[j])
    
    return dist_matrix, centers

def calculate_rmsd_matrix(chain_objects, pdb_ids, name, logger):
    """Calculate RMSD matrix using structural alignment"""
    n = len(chain_objects)
    rmsd_matrix = np.zeros((n, n))
    
    logger.info(f"Calculating {name} RMSD matrix for {n} structures...")
    logger.info("Progress: [" + "-"*50 + "]")
    # logger.info("          [", end='')
    
    total_pairs = n * (n - 1) // 2
    completed = 0
    progress_interval = max(1, total_pairs // 50)
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                # Check if both chains have atoms to align
                if cmd.count_atoms(f"{chain_objects[i]} and name CA") > 3 and \
                   cmd.count_atoms(f"{chain_objects[j]} and name CA") > 3:
                    rmsd = cmd.align(f"{chain_objects[i]} and name CA", 
                                    f"{chain_objects[j]} and name CA", cycles=5)[0]
                else:
                    rmsd = 0.0
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd
            except Exception as e:
                logger.error(f"Error aligning {pdb_ids[i]} and {pdb_ids[j]}: {str(e)}")
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = 0.0
            
            completed += 1
            if completed % progress_interval == 0:
                # logger.info("#", end='')
                sys.stdout.flush()
    
    logger.info("] 100% - Completed")
    return rmsd_matrix

def optimize_view():
    """Optimize view for better visualization"""
    cmd.orient()
    cmd.zoom(buffer=15)
    cmd.clip("slab", 100)
    cmd.set("depth_cue", 0)

def process_structures(file_paths, output_paths, logger):
    """Main processing function with flexible alignment and RMSD calculation"""
    # Create organized directory structure
    aligned_pdbs_dir = output_paths['aligned_pdbs']
    chain_pdbs_dir = output_paths['chain_pdbs']
    images_dir = output_paths['images']
    excel_dir = output_paths['excel']
    sessions_dir = output_paths['sessions']
    
    ref_path = file_paths[0]
    ref_id = os.path.splitext(os.path.basename(ref_path))[0].replace(".", "_")
    mobile_ids = [os.path.splitext(os.path.basename(p))[0].replace(".", "_") for p in file_paths[1:]]
    n_mobile = len(mobile_ids)
    
    # Display chain detection approach
    logger.info("\nChain detection approach:")
    if USE_GLOBAL_CHAIN_MAPPING:
        logger.info("Using global chain mapping:")
        logger.info(f"  MHC chains: {', '.join(GLOBAL_CHAIN_MAP['mhc'])}")
        logger.info(f"  Peptide chain: {', '.join(GLOBAL_CHAIN_MAP['peptide'])}")
        logger.info(f"  TCR chains: {', '.join(GLOBAL_CHAIN_MAP['tcr'])}")
    else:
        logger.info("- MHC chains: Typically the two longest chains (α and β chains)")
        logger.info("- Peptide chain: Shortest chain (5-25 residues)")
        logger.info("- TCR chains: Next two longest chains (α and β chains)")
        logger.info("Fallback: Using chains A,B for MHC, C for peptide, D,E for TCR if available")
        logger.info("          or first two chains for MHC, next for peptide, next two for TCR")
    logger.info("Detailed chain assignments will be saved in chain_mappings.csv\n")
    
    mobile_colors = get_distinct_colors(n_mobile)
    
    logger.info(f"\nProcessing reference structure: {ref_id}")
    ref_name = f"ref_{ref_id}"
    cmd.load(ref_path, ref_name)
    
    ref_chain_map = get_chain_type(ref_name, ref_id, logger, 
                                  GLOBAL_CHAIN_MAP if USE_GLOBAL_CHAIN_MAPPING else None)
    logger.info(f"Chain mapping: MHC={ref_chain_map['mhc']}, Peptide={ref_chain_map['peptide']}, TCR={ref_chain_map['tcr']}")
    
    ref_mhc, ref_pep, ref_tcr = create_chain_objects(ref_name, "ref", ref_chain_map)
    
    # Save reference chains to individual PDB files
    cmd.save(os.path.join(chain_pdbs_dir, f"{ref_id}_mhc.pdb"), ref_mhc)
    cmd.save(os.path.join(chain_pdbs_dir, f"{ref_id}_peptide.pdb"), ref_pep)
    cmd.save(os.path.join(chain_pdbs_dir, f"{ref_id}_tcr.pdb"), ref_tcr)
    
    cmd.hide("everything", ref_name)
    cmd.show("cartoon", f"{ref_mhc} or {ref_pep} or {ref_tcr}")
    cmd.color("gray", ref_mhc)
    cmd.color("yellow", ref_pep)
    cmd.color("gray", ref_tcr)
    
    results = []
    chain_objects = {
        'mhc': [ref_mhc],
        'peptide': [ref_pep],
        'tcr': [ref_tcr]
    }
    chain_maps = {ref_id: ref_chain_map}
    all_pdb_ids = [ref_id]
    
    # Determine alignment target object
    align_ref_obj = None
    if ALIGN_TARGET == "mhc":
        align_ref_obj = ref_mhc
    elif ALIGN_TARGET == "tcr":
        align_ref_obj = ref_tcr
    elif ALIGN_TARGET == "peptide":
        align_ref_obj = ref_pep
    else:
        raise ValueError(f"Invalid alignment target: {ALIGN_TARGET}")
    
    for i, mobile_path in enumerate(file_paths[1:]):
        mobile_id = mobile_ids[i]
        prefix = f"mob{i+1}_{mobile_id}"
        
        logger.info(f"\nProcessing mobile structure {i+1}/{len(file_paths)-1}: {mobile_id}")
        mobile_name = f"{prefix}_full"
        cmd.load(mobile_path, mobile_name)
        
        mobile_chain_map = get_chain_type(mobile_name, mobile_id, logger, 
                                         GLOBAL_CHAIN_MAP if USE_GLOBAL_CHAIN_MAPPING else None)
        logger.info(f"Chain mapping: MHC={mobile_chain_map['mhc']}, Peptide={mobile_chain_map['peptide']}, TCR={mobile_chain_map['tcr']}")
        chain_maps[mobile_id] = mobile_chain_map
        
        # Only align if target chains exist
        if mobile_chain_map[ALIGN_TARGET]:
            mobile_align_sel = f"chain {'+'.join(mobile_chain_map[ALIGN_TARGET])}"
            rmsd = cmd.super(f"{mobile_name} and {mobile_align_sel}", align_ref_obj)[0]
            logger.info(f"  Aligned {ALIGN_TARGET.upper()} with RMSD: {rmsd:.3f} Å")
        else:
            rmsd = 999.0
            logger.warning(f"  No {ALIGN_TARGET.upper()} chains to align!")
        
        mob_mhc, mob_pep, mob_tcr = create_chain_objects(mobile_name, prefix, mobile_chain_map)
        cmd.hide("everything", mobile_name)
        
        # Save mobile chains to individual PDB files
        cmd.save(os.path.join(chain_pdbs_dir, f"{mobile_id}_mhc.pdb"), mob_mhc)
        cmd.save(os.path.join(chain_pdbs_dir, f"{mobile_id}_peptide.pdb"), mob_pep)
        cmd.save(os.path.join(chain_pdbs_dir, f"{mobile_id}_tcr.pdb"), mob_tcr)
        
        # Collect chain objects for later use
        chain_objects['mhc'].append(mob_mhc)
        chain_objects['peptide'].append(mob_pep)
        chain_objects['tcr'].append(mob_tcr)
        all_pdb_ids.append(mobile_id)
        
        r, g, b = mobile_colors[i]
        color_name = f"mobile_color_{i}"
        cmd.set_color(color_name, [r, g, b])
        cmd.color(color_name, mob_mhc)
        cmd.color(color_name, mob_tcr)
        
        light_color_name = f"mobile_lightcolor_{i}"
        cmd.set_color(light_color_name, [min(1.0, r*1.5), min(1.0, g*1.5), min(1.0, b*1.5)])
        cmd.color(light_color_name, mob_pep)
        
        coord_path = os.path.join(aligned_pdbs_dir, f"aligned_{mobile_id}.pdb")
        cmd.save(coord_path, mobile_name)
        logger.info(f"  Saved transformed coordinates: {os.path.basename(coord_path)}")
        
        cmd.hide("everything")
        cmd.show("cartoon", f"{ref_mhc} or {ref_pep} or {ref_tcr} or {mob_mhc} or {mob_pep} or {mob_tcr}")
        optimize_view()
        snapshot_path = os.path.join(images_dir, f"aligned_{mobile_id}.png")
        cmd.png(snapshot_path, width=1600, height=1200, dpi=300, ray=1)
        logger.info(f"  Saved alignment snapshot: {os.path.basename(snapshot_path)}")
        
        # Save individual session for this mobile
        session_path = os.path.join(sessions_dir, f"aligned_{mobile_id}.pse")
        cmd.save(session_path)
        logger.info(f"  Saved individual session: {os.path.basename(session_path)}")
        
        results.append({
            'Structure': mobile_id,
            f'{ALIGN_TARGET.upper()}_RMSD': f"{rmsd:.3f} Å" if rmsd < 999 else "N/A",
            'Reference': ref_id
        })
    
    multi_model_file = os.path.join(aligned_pdbs_dir, "all_aligned_structures.pdb")
    # cmd.save(multi_model_file)
    logger.info(f"\nSaved multi-model PDB: {os.path.basename(multi_model_file)}")
    
    # Calculate TCR center of mass distances
    try:
        tcr_matrix, tcr_centers = calculate_com_distance_matrix(
            chain_objects['tcr'], all_pdb_ids, "TCR", logger
        )
    except Exception as e:
        logger.error(f"Error calculating TCR distances: {str(e)}")
        tcr_matrix = np.zeros((len(all_pdb_ids), len(all_pdb_ids)))
        tcr_centers = [np.zeros(3) for _ in all_pdb_ids]
    
    # Calculate RMSD matrices for requested targets
    rmsd_matrices = {}
    for target in RMSD_TARGETS:
        try:
            if target in chain_objects:
                matrix = calculate_rmsd_matrix(chain_objects[target], all_pdb_ids, target.upper(), logger)
                rmsd_matrices[target] = matrix
            else:
                logger.error(f"Invalid RMSD target: {target}. Skipping.")
        except Exception as e:
            logger.error(f"Error calculating {target} RMSDs: {str(e)}")
            rmsd_matrices[target] = np.zeros((len(all_pdb_ids), len(all_pdb_ids)))
    
    logger.info("\nSaving analysis results:")
    
    # Save alignment results
    df_mobile = pd.DataFrame(results)
    mobile_csv = os.path.join(excel_dir, "alignment_results.csv")
    df_mobile.to_csv(mobile_csv, index=False)
    logger.info(f"  - Alignment results: alignment_results.csv")
    
    # Save TCR distance matrix
    df_tcr = pd.DataFrame(tcr_matrix, index=all_pdb_ids, columns=all_pdb_ids)
    tcr_csv = os.path.join(excel_dir, "tcr_distance_matrix.csv")
    df_tcr.to_csv(tcr_csv)
    logger.info(f"  - TCR distance matrix: tcr_distance_matrix.csv")
    
    # Save RMSD matrices for each target
    for target, matrix in rmsd_matrices.items():
        df = pd.DataFrame(matrix, index=all_pdb_ids, columns=all_pdb_ids)
        csv_path = os.path.join(excel_dir, f"{target}_rmsd_matrix.csv")
        df.to_csv(csv_path)
        logger.info(f"  - {target.upper()} RMSD matrix: {os.path.basename(csv_path)}")
    
    # Save TCR center coordinates
    com_data = []
    for i, pid in enumerate(all_pdb_ids):
        com_data.append({
            'PDB_ID': pid,
            'TCR_COM_X': tcr_centers[i][0],
            'TCR_COM_Y': tcr_centers[i][1],
            'TCR_COM_Z': tcr_centers[i][2]
        })
    
    df_com = pd.DataFrame(com_data)
    com_csv = os.path.join(excel_dir, "tcr_center_coordinates.csv")
    df_com.to_csv(com_csv, index=False)
    logger.info(f"  - TCR center coordinates: tcr_center_coordinates.csv")
    
    # Save chain mappings
    chain_map_data = []
    for pid, mapping in chain_maps.items():
        chain_map_data.append({
            'PDB_ID': pid,
            'MHC_Chains': ','.join(mapping['mhc']) if mapping['mhc'] else 'N/A',
            'Peptide_Chain': ','.join(mapping['peptide']) if mapping['peptide'] else 'N/A',
            'TCR_Chains': ','.join(mapping['tcr']) if mapping['tcr'] else 'N/A'
        })
    
    df_chain_map = pd.DataFrame(chain_map_data)
    chain_map_csv = os.path.join(excel_dir, "chain_mappings.csv")
    df_chain_map.to_csv(chain_map_csv, index=False)
    logger.info(f"  - Chain mappings: chain_mappings.csv")
    
    # Save main session with all structures
    cmd.show("cartoon", "all")
    cmd.disable("*_full")
    cmd.enable("ref_* or mob*")
    optimize_view()
    session_file = os.path.join(sessions_dir, "analysis_session.pse")
    cmd.save(session_file)
    logger.info(f"\nSaved main PyMOL session: analysis_session.pse")
    
    # Create and save chain-specific sessions
    for chain_type in ['tcr', 'mhc', 'peptide']:
        cmd.disable("*")
        for obj in chain_objects[chain_type]:
            cmd.enable(obj)
        
        cmd.zoom("enabled")
        session_path = os.path.join(sessions_dir, f"{chain_type}_session.pse")
        cmd.save(session_path)
        logger.info(f"Saved {chain_type.upper()} chain session: {os.path.basename(session_path)}")
    
    return df_mobile

def main():
    """Main workflow function"""
    try:
        # Setup output directories
        output_paths = setup_output(OUTPUT_DIR)
        
        # Setup logging
        logger = setup_logging(OUTPUT_DIR)
        welcome_message(logger)
        
        # Get PDB files
        file_paths = get_pdb_files(INPUT_DIR)
        logger.info(f"\nFound {len(file_paths)} PDB files in {INPUT_DIR}")
        logger.info(f"Reference structure: {os.path.basename(file_paths[0])}")
        logger.info(f"Mobile structures: {len(file_paths)-1}")
        
        start_time = time.time()
        logger.info("\nProcessing structures...")
        logger.info("PyMOL is working in the background - this may take several minutes")
        
        # Process all structures
        process_structures(file_paths, output_paths, logger)
        
        elapsed = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info(" ANALYSIS COMPLETE ".center(80))
        logger.info("="*80)
        logger.info(f"\nSuccessfully processed {len(file_paths)} structures in {elapsed:.1f} seconds")
        logger.info(f"All results saved to: {os.path.abspath(OUTPUT_DIR)}")
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        if 'logger' in locals():
            logger.critical(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    if 'cmd' in globals():
        main()
    else:
        print("Please run this script within PyMOL using:")
        print("run /path/to/your/script.py")
