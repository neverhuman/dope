/* The child aborts on every panic; it has no unwinder or signal handlers.
 * Keep executable text/read-only constants RX, separate data RW, and NX stack.
 * No executable byte or input is excluded from the paid complete ELF. */
SECTIONS { /DISCARD/ : { *(.eh_frame) *(.eh_frame_hdr) *(.comment) } }
INSERT AFTER .bss;
