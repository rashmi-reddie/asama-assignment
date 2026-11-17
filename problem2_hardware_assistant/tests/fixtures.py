"""Captured-style outputs from real tools (trimmed). Used to test parsers on any OS."""

LSCPU = """Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
CPU(s):                               8
Thread(s) per core:                   2
Core(s) per socket:                   4
Socket(s):                            1
Vendor ID:                            GenuineIntel
Model name:                           11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
CPU max MHz:                          4200.0000
CPU min MHz:                          400.0000
L2 cache:                             5 MiB (4 instances)
L3 cache:                             8 MiB (1 instance)
Virtualization:                       VT-x
"""

DMIDECODE_ONE_FREE = """# dmidecode 3.3
Getting SMBIOS data from sysfs.

Handle 0x0031, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tError Correction Type: None
\tMaximum Capacity: 64 GB
\tError Information Handle: Not Provided
\tNumber Of Devices: 2

Handle 0x0032, DMI type 17, 40 bytes
Memory Device
\tArray Handle: 0x0031
\tSize: 8 GB
\tForm Factor: SODIMM
\tLocator: DIMM A
\tType: DDR4
\tSpeed: 3200 MT/s
\tManufacturer: Samsung
\tPart Number: M471A1K43DB1-CWE
\tConfigured Memory Speed: 3200 MT/s

Handle 0x0033, DMI type 17, 40 bytes
Memory Device
\tArray Handle: 0x0031
\tSize: No Module Installed
\tForm Factor: Unknown
\tLocator: DIMM B
\tType: Unknown
\tSpeed: Unknown
"""

DMIDECODE_FULL = DMIDECODE_ONE_FREE.replace("Size: No Module Installed\n\tForm Factor: Unknown", "Size: 8192 MB\n\tForm Factor: SODIMM")

DMIDECODE_SOLDERED = """Handle 0x0010, DMI type 16, 23 bytes
Physical Memory Array
\tMaximum Capacity: 16 GB
\tNumber Of Devices: 1

Handle 0x0011, DMI type 17, 40 bytes
Memory Device
\tSize: 16 GB
\tForm Factor: Row Of Chips
\tLocator: Onboard
\tType: LPDDR4X
\tSpeed: 4267 MT/s
"""

LSBLK = """{"blockdevices":[
 {"name":"nvme0n1","size":512110190592,"type":"disk","mountpoint":null,"model":"SAMSUNG MZVLQ512HBLU","rota":false,"tran":"nvme","fstype":null,
  "children":[{"name":"nvme0n1p1","size":536870912,"type":"part","mountpoint":"/boot/efi","model":null,"rota":false,"tran":null,"fstype":"vfat"},
              {"name":"nvme0n1p2","size":511573319680,"type":"part","mountpoint":"/","model":null,"rota":false,"tran":null,"fstype":"ext4"}]},
 {"name":"sda","size":1000204886016,"type":"disk","mountpoint":null,"model":"WDC WD10SPZX","rota":true,"tran":"sata","fstype":null},
 {"name":"zram0","size":8589934592,"type":"disk","mountpoint":"[SWAP]","model":null,"rota":false,"tran":null,"fstype":"swap"},
 {"name":"loop0","size":65011712,"type":"loop","mountpoint":"/snap/core","model":null,"rota":true,"tran":null,"fstype":"squashfs"}]}"""

LSPCI_MM = '''00:02.0 "VGA compatible controller" "Intel Corporation" "TigerLake-LP GT2 [Iris Xe Graphics]" -r01 "Dell" "Device 0a5e"
01:00.0 "3D controller" "NVIDIA Corporation" "GA107M [GeForce RTX 3050 Mobile]" -ra1 "Dell" "Device 0a5e"
00:1f.3 "Audio device" "Intel Corporation" "Tiger Lake-LP Smart Sound" -r20 "Dell" "Device 0a5e"
'''
NVIDIA_SMI = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 535.54.03\n"

FREE_B = """               total        used        free      shared  buff/cache   available
Mem:      8213000000  3000000000  2000000000   100000000  3213000000  4900000000
Swap:      2147479552           0  2147479552
"""

# ---- Windows: shapes produced by the PowerShell scripts in tools.py (ConvertTo-Json -Compress)
WIN_MEMORY_ONE = ('{"array":{"MaxCapacity":33554432,"MaxCapacityEx":33554432,"MemoryDevices":2},'
                  '"modules":{"DeviceLocator":"ChannelA-DIMM0","BankLabel":"BANK 0","Capacity":8589934592,"Speed":3200,'
                  '"ConfiguredClockSpeed":3200,"Manufacturer":"Micron","PartNumber":"4ATF1G64HZ-3G2E1","SMBIOSMemoryType":26,"FormFactor":12},'
                  '"os":{"TotalVisibleMemorySize":8118000,"FreePhysicalMemory":2400000}}')
WIN_MEMORY_TWO_LPDDR = ('{"array":[{"MaxCapacity":67108864,"MaxCapacityEx":67108864,"MemoryDevices":4}],'
                        '"modules":[{"DeviceLocator":"Controller0-ChannelA","Capacity":8589934592,"Speed":4267,"ConfiguredClockSpeed":4267,"Manufacturer":"Samsung","PartNumber":"K3KL","SMBIOSMemoryType":30,"FormFactor":0},'
                        '{"DeviceLocator":"Controller0-ChannelB","Capacity":8589934592,"Speed":4267,"ConfiguredClockSpeed":4267,"Manufacturer":"Samsung","PartNumber":"K3KL","SMBIOSMemoryType":30,"FormFactor":0}],'
                        '"os":{"TotalVisibleMemorySize":16000000,"FreePhysicalMemory":9000000}}')
WIN_CPU = ('{"Name":"AMD Ryzen 5 5500U with Radeon Graphics","Manufacturer":"AuthenticAMD","Architecture":9,"NumberOfCores":6,'
           '"NumberOfLogicalProcessors":12,"MaxClockSpeed":2100,"L2CacheSize":3072,"L3CacheSize":8192,"VirtualizationFirmwareEnabled":true}')
WIN_STORAGE = ('{"disks":{"DeviceId":"0","FriendlyName":"KBG40ZNS512G NVMe KIOXIA 512GB","MediaType":"SSD","BusType":"NVMe","Size":512110190592,"HealthStatus":"Healthy"},'
               '"volumes":[{"DeviceID":"C:","Size":511000000000,"FreeSpace":201000000000,"FileSystem":"NTFS"}]}')
WIN_GPU = ('{"controllers":[{"Name":"AMD Radeon(TM) Graphics","AdapterCompatibility":"Advanced Micro Devices, Inc.","AdapterRAM":536870912,"DriverVersion":"31.0.12044.4"},'
           '{"Name":"NVIDIA GeForce RTX 3050 Laptop GPU","AdapterCompatibility":"NVIDIA","AdapterRAM":4293918720,"DriverVersion":"31.0.15.3623"}],'
           '"nvidia_csv":"NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 536.23"}')
WIN_OVERVIEW = ('{"cs":{"Manufacturer":"HP","Model":"HP Laptop 15s-eq2xxx"},"os":{"Caption":"Microsoft Windows 11 Home","Version":"10.0.22631","OSArchitecture":"64-bit"},'
                '"bios":{"SMBIOSBIOSVersion":"F.22","ReleaseDate":"2023-05-10"}}')
WIN_BATTERY = ('{"battery":{"Name":"Primary","BatteryStatus":2,"EstimatedChargeRemaining":87},'
               '"full":{"FullChargedCapacity":36000},"design":{"DesignedCapacity":41000}}')
