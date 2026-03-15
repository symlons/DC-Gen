#!/bin/bash

DISKS=(
  "/tmp/testfile"
  "/mnt/Volume-eV4BofCN/testfile"
  "/mnt/SFS-iCxS1nYm/testfile"
)

for FILE in "${DISKS[@]}"; do
    DIR=$(dirname "$FILE")
    echo "Testing disk: $DIR"

    # Get device and filesystem info
    MOUNT_POINT=$(df "$DIR" | tail -1 | awk '{print $1}')
    FSTYPE=$(df -T "$DIR" | tail -1 | awk '{print $2}')

    NETWORK_SPEED=""
    if [[ "$FSTYPE" =~ nfs|cifs|smb3|sshfs ]]; then
        STORAGE_TYPE="Network ($FSTYPE)"
        # Measure effective network throughput
        WRITE_SPEED=$(sudo dd if=/dev/zero of="$FILE" bs=1G count=4 oflag=direct 2>&1 | grep -oP '\d+(\.\d+)? [MG]B/s')
        READ_SPEED=$(sudo dd if="$FILE" of=/dev/null bs=1G iflag=direct 2>&1 | grep -oP '\d+(\.\d+)? [MG]B/s')
        NETWORK_SPEED="Effective network speed: Write=$WRITE_SPEED, Read=$READ_SPEED"
    else
        # Local disk, check rotational
        DEVNAME=$(basename "$MOUNT_POINT")
        if [ -e "/sys/block/$DEVNAME/queue/rotational" ]; then
            ROTA=$(cat /sys/block/$DEVNAME/queue/rotational)
            if [ "$ROTA" -eq 0 ]; then
                STORAGE_TYPE="SSD/NVMe"
            else
                STORAGE_TYPE="HDD"
            fi
        else
            STORAGE_TYPE="Unknown"
        fi

        # Measure read/write speed for local disk
        WRITE_SPEED=$(sudo dd if=/dev/zero of="$FILE" bs=1G count=4 oflag=direct 2>&1 | grep -oP '\d+(\.\d+)? [MG]B/s')
        READ_SPEED=$(sudo dd if="$FILE" of=/dev/null bs=1G iflag=direct 2>&1 | grep -oP '\d+(\.\d+)? [MG]B/s')
    fi

    echo "  Storage type: $STORAGE_TYPE"
    echo "  Write speed: $WRITE_SPEED"
    echo "  Read speed:  $READ_SPEED"
    if [ -n "$NETWORK_SPEED" ]; then
        echo "  $NETWORK_SPEED"
    fi

    rm -f "$FILE"
done

