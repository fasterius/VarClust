import os
import vcf
import pandas as pd


# Creation of profiles


def profile_sort_order(string):
    "Sort order specification."

    s = string.split('\t')
    return [s[0].upper().replace('_', ' '),
            int(s[1]),
            s[3].upper(),
            s[4].upper(),
            s[5].upper().replace('.', ':'),
            s[8].upper(),
            s[9].upper(),
            s[10].upper()]


def full_profile(chrom, pos, rsid, ref, alt, dp, ad, A1GT, A2GT, ann):
    "Function for extracting and prioritising per-variant impacts."

    # Collate record info to a set
    result = ['%s %s %s %s %s %s %s %s %s %s' %
              (line.split('|')[3], line.split('|')[4], line.split('|')[2],
               line.split('|')[1], line.split('|')[5], line.split('|')[6],
               line.split('|')[7], line.split('|')[9], line.split('|')[10],
               line.split('|')[15]) for line in ann]
    result = set(result)

    # Priority list for impacts
    priority_list = ['MODIFIER', 'LOW', 'MODERATE', 'HIGH']

    # Find the highest impact for the current record
    max_index = -1
    for line in result:
        [gene, ensgid, impact, effect, feature, enst, biotype, nucl, aacid,
            warnings] = line.split(' ')
        max_index = max(max_index, priority_list.index(impact))
    max_impact = priority_list[max_index]

    # Add all unique highest impact lines to final set
    unique_lines = set()
    for line in result:
        [gene, ensgid, impact, effect, feature, enst, biotype, nucl, aacid,
            warnings] = line.split(' ')
        if impact == max_impact:
            current = str(chrom) + "\t" + \
                      str(pos) + "\t" + \
                      str(rsid) + "\t" +  \
                      str(gene) + "\t" + \
                      str(ensgid) + "\t" + \
                      str(enst) + "\t" + \
                      str(ref) + "\t" + \
                      str(alt[0]) + "\t" + \
                      str(impact) + "\t" + \
                      str(effect) + "\t" + \
                      str(feature) + "\t" + \
                      str(biotype) + "\t" + \
                      str(dp) + "\t" + \
                      str(ad[0]) + "\t" + \
                      str(ad[1]) + "\t" + \
                      str(A1GT) + "\t" + \
                      str(A2GT) + "\t" + \
                      str(warnings) + "\n"

            # Add to final set (if unique)
            if current not in unique_lines:
                unique_lines.add(current)

    return unique_lines


def create_profile(input_file,
                   input_sample,
                   output_file,
                   filter_depth=10,
                   method="full"):
    "Create SNV profiles from VCF files."

    # Remove output file if already existing
    if os.path.isfile(output_file):
        os.remove(output_file)

    # Open output file for appending
    output_file = open(output_file, 'a')

    # Header row
    if method == "position_only":
        header = \
            'chr\t' + \
            'pos\t' + \
            'DP\t' + \
            'AD1\t' + \
            'AD2\t' + \
            'A1\t' + \
            'A2\n'
    elif method == "full":
        header = \
            'chr\t' + \
            'pos\t' + \
            'rsID\t' + \
            'gene\t' + \
            'ENSGID\t' + \
            'ENSTID\t' + \
            'REF\t' + \
            'ALT\t' + \
            'impact\t' + \
            'effect\t' + \
            'feature\t' + \
            'biotype\t' + \
            'DP\t' + \
            'AD1\t' + \
            'AD2\t' + \
            'A1\t' + \
            'A2\t' + \
            'warnings\n'
    else:
        raise ValueError('wrong method specification \"' + method + '\"; ' +
                         'please use <full> or <position_only>')

    # Write header to file
    output_file.write(header)

    # Open input VCF file
    vcf_reader = vcf.Reader(filename=input_file)

    # Read each record in the VCF file
    for record in vcf_reader:

        # Get record info
        ref = record.REF
        alt = str(record.ALT).strip('[]')
        chrom = record.CHROM
        pos = record.POS
        if method == 'full':
            rsid = record.ID

        # Skip non-SNVs
        if len(ref) > 1 or len(alt) > 1:
            continue

        # Collect genotype info (skip record if any is missing)
        try:
            gt = record.genotype(str(input_sample))['GT']
            ad = record.genotype(str(input_sample))['AD']
            dp = record.genotype(str(input_sample))['DP']
        except AttributeError:
            continue

        # Skip record if no call was made
        if gt is None:
            continue

        # Collect annotation information (skip record if missing)
        try:
            filt = record.FILTER
            if method == 'full':
                ann = record.INFO['ANN']
        except KeyError:
            continue

        # Skip variant if filtering depth is below threshold
        if dp < filter_depth:
            continue

        # Get filter info
        if filt:
            filt = filt[0]
        else:
            filt = 'None'

        # Skip record if it doesn't pass filters
        if filt != 'None':
            continue

        # Make AD into list if only one value is available
        try:
            ad[0] = ad[0]
        except TypeError:
            ad = [ad, 0]

        # Get genotypes
        gts = gt.split('/')
        A1 = gts[0]
        A2 = gts[1]

        # First allele
        if A1 == '0':
            A1GT = ref
        else:
            A1GT = alt

        # Second allele
        if A2 == '0':
            A2GT = ref
        else:
            A2GT = alt

        # Finalise current variant output
        if method == "full":
            output = full_profile(chrom, pos, rsid, ref, alt, dp, ad, A1GT,
                                  A2GT, ann)
            output_file.writelines(sorted(output, key=profile_sort_order))

        elif method == "position_only":
            output = \
                str(chrom) + '\t' + \
                str(pos) + '\t' + \
                str(dp) + '\t' + \
                str(ad[0]) + '\t' + \
                str(ad[1]) + '\t' + \
                str(A1GT) + '\t' + \
                str(A2GT) + '\n'
            output_file.write(output)

    # Close output file
    output_file.close()


def create_profiles_in_dir(input_dir,
                           output_dir,
                           filter_depth=10,
                           method='full'):
    "Creates profiles for each VCF in a directory."

    # Get all files in directory
    files = os.listdir(input_dir)

    # Only use VCFs
    files = [name for name in files if 'vcf' in name]

    # Calculate total number of profiles and initialise counter
    nn_tot = len(files)
    nn = 1

    # Loop through each VCF
    for file in files:

        # Get current input_sample name
        input_sample = file.replace('.vcf.gz', '')

        # Fix variables and filenames
        profile = file.replace('.vcf.gz', '.profile.txt')
        output_file = output_dir + '/' + profile
        file = input_dir + '/' + file

        # Create SNV profile from VCF
        print('Creating profile for ' + input_sample + ' [' + str(nn) + ' / ' +
              str(nn_tot) + ']')
        create_profile(file, input_sample, output_file, filter_depth, method)

        # Increment counter
        nn += 1


# Reading of profiles


def sort_genotype(row_data):
    "Converts genotypes into 10-element vectors"

    # Sort the genotype column
    sorted_genotype = ''.join(sorted(row_data['genotype']))
    row_data['genotype'] = sorted_genotype

    # Return row_data with sorted genotype
    return row_data


def read_profile(file,
                 merge_by='gene',
                 subset_cols=None,
                 subset_values=None):
    """Reads a SNV profile."""

    # Read file with pandas
    data = pd.read_csv(file, sep='\t')

    # Drop non-standard chromosomes
    standard = ['chr' + str(chr) for chr in range(1, 23)]
    standard.extend(['chrX', 'chrY', 'chrMT'])
    data = data.loc[data['chr'].isin(standard)]

    # Subset (if applicable)
    if subset_cols is not None:
        if subset_values is not None:
            for col in subset_cols:
                data = data.loc[data[col].isin(subset_values)]

    # Remove duplicated variants at specified level (if present)
    if merge_by == 'gene':
        unique_level = ['chr', 'pos', 'ENSGID']
    elif merge_by == 'position':
        unique_level = ['chr', 'pos']
    else:
        raise ValueError('unique variant level specification \"' + merge_by +
                         '\" not valid; use \"gene\" or \"position\".')
    data = data.drop_duplicates(subset=unique_level)

    # Merge and sort alleles into genotypes (if not already done)
    if 'genotype' not in data.columns:
        data['genotype'] = data['A1'] + data['A2']
        data = data.apply(sort_genotype, axis=1)
        data = data.drop(columns=['A1', 'A2'])

    # Return final dataframe
    return data


def read_profile_dir(in_dir,
                     merge_by='gene',
                     samples=None,
                     subset_cols=None,
                     subset_values=None):
    "Reads all SNV profiles in a given directory"

    # List all profile files
    files = [os.path.join(in_dir, file) for file in os.listdir(in_dir)]
    files = [name for name in files if '.profile.txt' in name]

    # Calculate total profiles to read and initialise counter
    nn_tot = len(files) if samples is None else len(samples)
    nn = 1

    # Read each profile and save in a dictionary
    profiles = {}
    for file in files:

        # Get current sample name
        sample = file.split('/')[-1].split('.')[0]

        # Only read provided samples (if applicable)
        if samples is not None:
            if sample not in samples:
                continue

        # Read profile and add to dictionary
        print('Reading profile for ' + sample + ' [' + str(nn) + ' / ' +
              str(nn_tot) + ']')
        profiles[sample] = read_profile(file, merge_by, subset_cols,
                                        subset_values)

        # Increment counter
        nn += 1

    # Return profiles
    print('Done.')
    return profiles


# Aggregation of profiles


def overlap_profiles(pseudo,
                     profile,
                     merge_by='gene',
                     merge_method='outer'):
    """
    Overlaps and merge a pseudo-profiles and an SNV profile by the union of
    their matching SNVs.
    """

    # Check that `pseudo` contains a <count> column
    if 'count' not in pseudo.columns:
        raise ValueError('\"count\" column not present in provided ' +
                         'pseudo-profile')

    # Get columns to merge on
    if merge_by == 'gene':
        merge_cols = ['chr', 'pos', 'ENSGID', 'genotype']
    elif merge_by == 'position':
        merge_cols = ['chr', 'pos', 'genotype']
    else:
        raise ValueError('merge level specification \"' + merge_by +
                         '\" invalid; use \"gene\" or \"position\"')

    # Find available meta-columns
    all_cols = ['rsID', 'gene', 'impact', 'effect', 'feature', 'biotype']
    meta_cols = [x for x in all_cols if x in frozenset(pseudo.columns)]

    # Merge profiles
    pseudo = pd.merge(pseudo, profile, on=merge_cols, how=merge_method)

    #  # Remove non-matching genotypes
    #  pseudo = pseudo.loc[(pseudo['genotype_x'] == pseudo['genotype_y']) |
                        #  pseudo['genotype_x'].isnull() |
                        #  pseudo['genotype_y'].isnull()]

    # Increase variant counter
    pseudo.loc[pseudo['rsID_x'].notnull() &
               pseudo['rsID_y'].notnull(), 'count'] += 1
    pseudo.loc[pseudo['rsID_x'].isnull() &
               pseudo['rsID_y'].notnull(), 'count'] = 1

    # Move profile-unique rows to the pseudo columns
    for col in meta_cols:
        pseudo.loc[pseudo[col + '_x'].isnull(), col + '_x'] = \
            pseudo.loc[pseudo[col + '_x'].isnull(), col + '_y']

    # Remove redundant columns
    pseudo = pseudo[merge_cols + [col + '_x' for col in meta_cols] + ['count']]
    pseudo.columns = merge_cols + meta_cols + ['count']
    pseudo = pseudo.drop_duplicates(subset=merge_cols)

    # Return pseudoprofile
    return pseudo


def create_pseudo_profile(input_dir,
                          output_file,
                          metadata=None,
                          metadata_id_col=None,
                          sample_subset_cols=None,
                          sample_subset_values=None,
                          variant_subset_cols=None,
                          variant_subset_values=None,
                          merge_by='gene',
                          merge_method='inner',
                          cutoff=0,
                          encoding='iso8859_16'):
    """
    Create a pseudo-profile of all the SNV profiles given in the input
    directory.
    """

    # Read, subset and group on metadata or read all profiles
    if metadata is not None:

        # Check for metadata ID column
        if metadata_id_col is None:
            raise ValueError('Please provide a metadata ID column.')

        # Read metadata
        metadata = pd.read_table(metadata, encoding=encoding)

        # Subset metadata (if applicable)
        if sample_subset_cols is not None and sample_subset_values is not None:
            mask = metadata[sample_subset_cols].isin(sample_subset_values)\
                    .all(axis=1)
            metadata = metadata.loc[mask]
        elif (sample_subset_cols is None and
              sample_subset_values is not None) or \
             (sample_subset_cols is not None and
              sample_subset_values is None):
            raise ValueError('Please provide both column and values to subset '
                             'on if subsetting is desired.')

        # Read profile subset
        sample_subset = metadata[metadata_id_col].tolist()

    else:

        # No sample subsetting
        sample_subset = None

    # Read profiles
    profiles = read_profile_dir(input_dir,
                                merge_by,
                                sample_subset,
                                variant_subset_cols,
                                variant_subset_values)

    # Initialise pseudo-profile with the first sample
    print('Creating pseudo-profile ...')
    samples = sorted(list(profiles.keys()))
    pseudo = profiles[samples[0]]
    pseudo['count'] = 1

    # Create pseudo-profile
    nn = 1
    for sample in samples[1:len(samples)]:
        print('Overlapping sample ' + str(nn) + ' / ' + str(len(profiles)-1))
        pseudo = overlap_profiles(pseudo,
                                  profiles[sample],
                                  merge_by,
                                  merge_method)
        nn += 1

    # Add proportions of each variant in final pseudo-profile
    pseudo['proportion'] = round(pseudo['count'] / len(profiles) * 100, 1)

    # Remove variants/genes below proportion cutoff
    pseudo = pseudo.loc[pseudo['proportion'] >= cutoff, ]

    # Sort and write to file
    pseudo = pseudo.sort_values(by='count', axis=0, ascending=False)
    pseudo.to_csv(output_file, sep='\t', index=False)
    print('Done.')
