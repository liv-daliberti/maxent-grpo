#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef __NR_landlock_create_ruleset
#error "Landlock syscall numbers are required"
#endif

#define EXIT_SANDBOX_SETUP 125
#define MINIMUM_LANDLOCK_ABI 5

static void die(const char *message) {
    perror(message);
    exit(EXIT_SANDBOX_SETUP);
}

static void die_message(const char *message) {
    fprintf(stderr, "constructive_code_sandbox: %s\n", message);
    exit(EXIT_SANDBOX_SETUP);
}

static unsigned long long parse_limit(const char *raw, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value == 0) {
        fprintf(stderr, "constructive_code_sandbox: invalid %s: %s\n", label, raw);
        exit(EXIT_SANDBOX_SETUP);
    }
    return value;
}

static void set_limit(int resource, rlim_t soft, rlim_t hard, const char *label) {
    struct rlimit limit = {.rlim_cur = soft, .rlim_max = hard};
    if (setrlimit(resource, &limit) != 0) {
        die(label);
    }
}

static int landlock_create_ruleset(
    const struct landlock_ruleset_attr *attr,
    size_t size,
    uint32_t flags
) {
    return (int)syscall(__NR_landlock_create_ruleset, attr, size, flags);
}

static int landlock_add_rule(
    int ruleset_fd,
    enum landlock_rule_type type,
    const void *attr,
    uint32_t flags
) {
    return (int)syscall(__NR_landlock_add_rule, ruleset_fd, type, attr, flags);
}

static int landlock_restrict_self(int ruleset_fd, uint32_t flags) {
    return (int)syscall(__NR_landlock_restrict_self, ruleset_fd, flags);
}

static void add_path_rule(int ruleset_fd, const char *path, uint64_t access) {
    int path_fd = open(path, O_PATH | O_CLOEXEC);
    if (path_fd < 0) {
        die(path);
    }
    struct landlock_path_beneath_attr rule = {
        .allowed_access = access,
        .parent_fd = path_fd,
    };
    if (landlock_add_rule(ruleset_fd, LANDLOCK_RULE_PATH_BENEATH, &rule, 0) != 0) {
        close(path_fd);
        die("landlock_add_rule");
    }
    close(path_fd);
}

static void install_landlock(const char *image_root, const char *workdir) {
    int abi = landlock_create_ruleset(
        NULL, 0, LANDLOCK_CREATE_RULESET_VERSION
    );
    if (abi < MINIMUM_LANDLOCK_ABI) {
        die_message("Landlock ABI 5 or newer is required");
    }

    const uint64_t read_execute =
        LANDLOCK_ACCESS_FS_EXECUTE |
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_READ_DIR;
    const uint64_t writable_workdir =
        LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_READ_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_FILE |
        LANDLOCK_ACCESS_FS_MAKE_DIR |
        LANDLOCK_ACCESS_FS_MAKE_REG |
        LANDLOCK_ACCESS_FS_MAKE_SOCK |
        LANDLOCK_ACCESS_FS_MAKE_FIFO |
        LANDLOCK_ACCESS_FS_MAKE_SYM |
        LANDLOCK_ACCESS_FS_REFER |
        LANDLOCK_ACCESS_FS_TRUNCATE;
    const uint64_t handled =
        read_execute |
        writable_workdir |
        LANDLOCK_ACCESS_FS_MAKE_CHAR |
        LANDLOCK_ACCESS_FS_MAKE_BLOCK |
        LANDLOCK_ACCESS_FS_IOCTL_DEV;
    const struct landlock_ruleset_attr ruleset = {
        .handled_access_fs = handled,
        .handled_access_net =
            LANDLOCK_ACCESS_NET_BIND_TCP |
            LANDLOCK_ACCESS_NET_CONNECT_TCP,
    };
    int ruleset_fd = landlock_create_ruleset(&ruleset, sizeof(ruleset), 0);
    if (ruleset_fd < 0) {
        die("landlock_create_ruleset");
    }

    add_path_rule(ruleset_fd, image_root, read_execute);
    add_path_rule(ruleset_fd, workdir, writable_workdir);
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        close(ruleset_fd);
        die("PR_SET_NO_NEW_PRIVS");
    }
    if (landlock_restrict_self(ruleset_fd, 0) != 0) {
        close(ruleset_fd);
        die("landlock_restrict_self");
    }
    close(ruleset_fd);
}

#define DENY_SYSCALL(name) \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_##name, 0, 1), \
    BPF_STMT( \
        BPF_RET | BPF_K, \
        SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA) \
    )

static void install_seccomp(void) {
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_prlimit64, 0, 4),
        BPF_STMT(
            BPF_LD | BPF_W | BPF_ABS,
            offsetof(struct seccomp_data, args[0])
        ),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(
            BPF_RET | BPF_K,
            SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA)
        ),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        DENY_SYSCALL(socket),
        DENY_SYSCALL(socketpair),
        DENY_SYSCALL(connect),
        DENY_SYSCALL(accept),
        DENY_SYSCALL(accept4),
        DENY_SYSCALL(bind),
        DENY_SYSCALL(listen),
        DENY_SYSCALL(sendto),
        DENY_SYSCALL(sendmsg),
        DENY_SYSCALL(sendmmsg),
        DENY_SYSCALL(recvfrom),
        DENY_SYSCALL(recvmsg),
        DENY_SYSCALL(recvmmsg),
        DENY_SYSCALL(shutdown),
        DENY_SYSCALL(getsockname),
        DENY_SYSCALL(getpeername),
        DENY_SYSCALL(setsockopt),
        DENY_SYSCALL(getsockopt),
        DENY_SYSCALL(clone),
        DENY_SYSCALL(clone3),
        DENY_SYSCALL(fork),
        DENY_SYSCALL(vfork),
        DENY_SYSCALL(execveat),
        DENY_SYSCALL(kill),
        DENY_SYSCALL(tkill),
        DENY_SYSCALL(tgkill),
        DENY_SYSCALL(rt_sigqueueinfo),
        DENY_SYSCALL(rt_tgsigqueueinfo),
        DENY_SYSCALL(pidfd_open),
        DENY_SYSCALL(pidfd_getfd),
        DENY_SYSCALL(pidfd_send_signal),
        DENY_SYSCALL(ptrace),
        DENY_SYSCALL(process_vm_readv),
        DENY_SYSCALL(process_vm_writev),
        DENY_SYSCALL(process_madvise),
        DENY_SYSCALL(process_mrelease),
        DENY_SYSCALL(kcmp),
        DENY_SYSCALL(mount),
        DENY_SYSCALL(umount2),
        DENY_SYSCALL(pivot_root),
        DENY_SYSCALL(chroot),
        DENY_SYSCALL(unshare),
        DENY_SYSCALL(setns),
        DENY_SYSCALL(open_by_handle_at),
        DENY_SYSCALL(name_to_handle_at),
        DENY_SYSCALL(bpf),
        DENY_SYSCALL(perf_event_open),
        DENY_SYSCALL(userfaultfd),
        DENY_SYSCALL(io_uring_setup),
        DENY_SYSCALL(io_uring_enter),
        DENY_SYSCALL(io_uring_register),
        DENY_SYSCALL(fanotify_init),
        DENY_SYSCALL(fanotify_mark),
        DENY_SYSCALL(keyctl),
        DENY_SYSCALL(add_key),
        DENY_SYSCALL(request_key),
        DENY_SYSCALL(init_module),
        DENY_SYSCALL(finit_module),
        DENY_SYSCALL(delete_module),
        DENY_SYSCALL(kexec_load),
        DENY_SYSCALL(kexec_file_load),
        DENY_SYSCALL(reboot),
        DENY_SYSCALL(swapon),
        DENY_SYSCALL(swapoff),
        DENY_SYSCALL(acct),
        DENY_SYSCALL(quotactl),
        DENY_SYSCALL(syslog),
        DENY_SYSCALL(sethostname),
        DENY_SYSCALL(setdomainname),
        DENY_SYSCALL(setuid),
        DENY_SYSCALL(setgid),
        DENY_SYSCALL(setreuid),
        DENY_SYSCALL(setregid),
        DENY_SYSCALL(setresuid),
        DENY_SYSCALL(setresgid),
        DENY_SYSCALL(setfsuid),
        DENY_SYSCALL(setfsgid),
        DENY_SYSCALL(setgroups),
        DENY_SYSCALL(capset),
        DENY_SYSCALL(personality),
        DENY_SYSCALL(modify_ldt),
        DENY_SYSCALL(iopl),
        DENY_SYSCALL(ioperm),
        DENY_SYSCALL(mknod),
        DENY_SYSCALL(mknodat),
        DENY_SYSCALL(shmget),
        DENY_SYSCALL(shmat),
        DENY_SYSCALL(shmdt),
        DENY_SYSCALL(shmctl),
        DENY_SYSCALL(semget),
        DENY_SYSCALL(semop),
        DENY_SYSCALL(semtimedop),
        DENY_SYSCALL(semctl),
        DENY_SYSCALL(msgget),
        DENY_SYSCALL(msgsnd),
        DENY_SYSCALL(msgrcv),
        DENY_SYSCALL(msgctl),
        DENY_SYSCALL(setpriority),
        DENY_SYSCALL(sched_setaffinity),
        DENY_SYSCALL(sched_setparam),
        DENY_SYSCALL(sched_setscheduler),
        DENY_SYSCALL(sched_setattr),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    const struct sock_fprog program = {
        .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
        .filter = filter,
    };
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
        die("PR_SET_SECCOMP");
    }
}

static char *resolve_directory(const char *raw, const char *label) {
    char *resolved = realpath(raw, NULL);
    struct stat status;
    if (resolved == NULL || stat(resolved, &status) != 0 || !S_ISDIR(status.st_mode)) {
        free(resolved);
        die_message(label);
    }
    return resolved;
}

static void close_extra_file_descriptors(void) {
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
        die("getrlimit(RLIMIT_NOFILE)");
    }
    rlim_t maximum = limit.rlim_cur;
    if (maximum == RLIM_INFINITY || maximum > 65536) {
        maximum = 65536;
    }
    for (int fd = 3; (rlim_t)fd < maximum; ++fd) {
        close(fd);
    }
}

int main(int argc, char **argv) {
    if (argc != 8) {
        die_message(
            "usage: sandbox IMAGE_ROOT WORKDIR PROGRAM CPU_SECONDS "
            "MEMORY_BYTES OUTPUT_BYTES FILE_COUNT"
        );
    }
    if (strchr(argv[3], '/') != NULL || strcmp(argv[3], ".") == 0 ||
        strcmp(argv[3], "..") == 0) {
        die_message("PROGRAM must be a basename inside WORKDIR");
    }

    char *image_root = resolve_directory(argv[1], "invalid IMAGE_ROOT");
    char *workdir = resolve_directory(argv[2], "invalid WORKDIR");
    const unsigned long long cpu_seconds = parse_limit(argv[4], "CPU_SECONDS");
    const unsigned long long memory_bytes = parse_limit(argv[5], "MEMORY_BYTES");
    const unsigned long long output_bytes = parse_limit(argv[6], "OUTPUT_BYTES");
    const unsigned long long file_count = parse_limit(argv[7], "FILE_COUNT");

    char program_path[4096];
    char loader_path[4096];
    char python_path[4096];
    char python_home[4096];
    char library_path[12288];
    if (snprintf(program_path, sizeof(program_path), "%s/%s", workdir, argv[3]) >=
            (int)sizeof(program_path) ||
        snprintf(loader_path, sizeof(loader_path),
                 "%s/usr/lib64/ld-linux-x86-64.so.2", image_root) >=
            (int)sizeof(loader_path) ||
        snprintf(python_path, sizeof(python_path),
                 "%s/usr/local/bin/python3.10", image_root) >=
            (int)sizeof(python_path) ||
        snprintf(python_home, sizeof(python_home), "%s/usr/local", image_root) >=
            (int)sizeof(python_home) ||
        snprintf(library_path, sizeof(library_path),
                 "%s/usr/local/lib:%s/usr/lib/x86_64-linux-gnu:"
                 "%s/lib/x86_64-linux-gnu", image_root, image_root, image_root) >=
            (int)sizeof(library_path)) {
        die_message("runtime path is too long");
    }
    struct stat status;
    if (stat(program_path, &status) != 0 || !S_ISREG(status.st_mode)) {
        die_message("PROGRAM is not a regular file inside WORKDIR");
    }
    if (access(loader_path, X_OK) != 0 || access(python_path, X_OK) != 0) {
        die_message("pinned loader or Python executable is missing");
    }
    if (chdir(workdir) != 0) {
        die("chdir(WORKDIR)");
    }

    close_extra_file_descriptors();
    set_limit(RLIMIT_CPU, (rlim_t)cpu_seconds, (rlim_t)cpu_seconds,
              "setrlimit(RLIMIT_CPU)");
    set_limit(RLIMIT_AS, (rlim_t)memory_bytes, (rlim_t)memory_bytes,
              "setrlimit(RLIMIT_AS)");
    set_limit(RLIMIT_FSIZE, (rlim_t)output_bytes, (rlim_t)output_bytes,
              "setrlimit(RLIMIT_FSIZE)");
    set_limit(RLIMIT_NOFILE, (rlim_t)file_count, (rlim_t)file_count,
              "setrlimit(RLIMIT_NOFILE)");
    set_limit(RLIMIT_NPROC, 1, 1, "setrlimit(RLIMIT_NPROC)");
    set_limit(RLIMIT_CORE, 0, 0, "setrlimit(RLIMIT_CORE)");
    umask(077);

    install_landlock(image_root, workdir);
    install_seccomp();

    if (clearenv() != 0 ||
        setenv("PYTHONHOME", python_home, 1) != 0 ||
        setenv("PYTHONHASHSEED", "0", 1) != 0 ||
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1) != 0 ||
        setenv("LC_ALL", "C.UTF-8", 1) != 0 ||
        setenv("LANG", "C.UTF-8", 1) != 0 ||
        setenv("TZ", "UTC", 1) != 0 ||
        setenv("TMPDIR", workdir, 1) != 0 ||
        setenv("PATH", "", 1) != 0) {
        die("clean environment");
    }

    char *const python_argv[] = {
        loader_path,
        "--library-path",
        library_path,
        python_path,
        "-I",
        "-B",
        program_path,
        NULL,
    };
    execve(loader_path, python_argv, environ);
    die("execve(pinned Python)");
}
