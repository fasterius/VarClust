import os
import vcf


def col_sort(string):
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
            output_file.writelines(sorted(output, key=col_sort))

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
