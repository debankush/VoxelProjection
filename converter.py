#A script to convert degrees to decimals.
deg_in = input('Enter the values as: units mins secs ')
def deci_conv(units, mins, secs):
    sign = -1 if units.strip().startswith('-') else 1
    units = abs(float(units))
    return sign * (units + float(mins)/60 + float(secs)/3600)

parts = deg_in.split()
if len(parts) != 3:
    raise ValueError
deci_val = deci_conv(parts[0], parts[1], parts[2])
print(f"Decimal Value: {deci_val:.6f}")