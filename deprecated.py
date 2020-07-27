
def plotter(field, history, intrinsic, dt, threshold=0.2, min_time=5e-3, padding=1e-3):
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 22})

    eq_pa_plots  = eq_pitch_angle_from_moment(history, intrinsic)
    eq_pa_values = get_eq_pas(field, history, intrinsic, threshold)
    pa           = pitch_angle(history)
    K            = kinetic_energy(history, intrinsic)
    v_pa, v_pam  = velocity_par(history)
    v_pe, v_pem  = velocity_perp(history)
    b            = b_mag(history) * 1e9
    r            = position(history)
    r_mag        = position_mag(history)
    gr           = gyrorad(history, intrinsic)
    gf           = gyrofreq(history, intrinsic)
    
    num_particles = len(history[:, 0, 0, 0])
    steps         = len(history[0, :, 0, 0])
    t_v          = np.arange(0, steps) * dt
    
    def plot(particle_ind):
        if type(particle_ind) != list:
            particle_ind = [particle_ind]
            
        n = len(particle_ind)
        
        fig = plt.figure(figsize=(20, 40))
        gs = GridSpec(10, 10, figure=fig)
        
        ax10 = fig.add_subplot(gs[0, 0:3])
        for i, j in enumerate(particle_ind):
            ax10.plot(r[j, :, 0] * inv_Re, r[j, :, 2] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax10.set_xlabel(r'$x_{GSM}$ ($R_E$)')
        ax10.set_ylabel(r'$z_{GSM}$ ($R_E$)')
        ax10.grid()
        
        ax11 = fig.add_subplot(gs[0, 3:6])
        for i, j in enumerate(particle_ind):
            ax11.plot(r[j, :, 0] * inv_Re, r[j, :, 1] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax11.set_xlabel(r'$x_{GSM}$ ($R_E$)')
        ax11.set_ylabel(r'$y_{GSM}$ ($R_E$)')
        ax11.grid()
        
        ax12 = fig.add_subplot(gs[0, 6:9])
        for i, j in enumerate(particle_ind):
            ax12.plot(r[j, :, 1] * inv_Re, r[j, :, 2] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax12.set_xlabel(r'$y_{GSM}$ ($R_E$)')
        ax12.set_ylabel(r'$z_{GSM}$ ($R_E$)')
        ax12.grid()
        
        ax1 = fig.add_subplot(gs[1, :])
        for i, j in enumerate(particle_ind):
            ax1.plot(t_v, eq_pa_plots[j, :], zorder=1, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
            for k in range(len(eq_pa_values[j, :, 0])):
                value   = eq_pa_values[j, k, 0]
                l_point = eq_pa_values[j, k, 1]
                r_point = eq_pa_values[j, k, 2]
                if value != -1.0:
                    ax1.hlines(value, l_point * dt, r_point * dt, zorder=2, linewidth=5, linestyle=':', colors=plt.cm.magma(i / n))
        ax1.set_xlim([0, steps * dt])
        ax1.set_ylim([0, 90])
        ax1.set_ylabel('Equatorial\nPitch angle (deg)')
        ax1.grid()
        
        ax2 = fig.add_subplot(gs[2, :])
        for i, j in enumerate(particle_ind):
            ax2.plot(t_v, pa[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax2.set_xlim([0, steps * dt])
        ax2.set_ylim([0, 180])
        ax2.set_ylabel('Pitch angle (deg)')
        ax2.grid()
        
        ax3 = fig.add_subplot(gs[3, :])
        for i, j in enumerate(particle_ind):
            ax3.plot(t_v, r_mag[j, :] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax3.set_xlim([0, steps * dt])
        ax3.set_ylim([0, np.amax(r_mag[particle_ind, :]) * inv_Re])
        ax3.set_ylabel('Distance from\nGSM origin ($R_E$)')
        ax3.grid()
        
        ax4 = fig.add_subplot(gs[4, :])
        for i, j in enumerate(particle_ind):
            ax4.plot(t_v, K[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax4.set_xlim([0, steps * dt])
        ax4.set_ylim([np.amin(K[particle_ind, 0]) * (1 - 1e-3), np.amax(K[particle_ind, 0]) * (1 + 1e-3)])
        ax4.set_ylabel('Energy (eV)')
        ax4.grid()
        
        ax5 = fig.add_subplot(gs[5, :])
        for i, j in enumerate(particle_ind):
            ax5.plot(t_v, b[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax5.set_xlim([0, steps * dt])
        ax5.set_ylim([np.amin(b[particle_ind, :]) * (1 - 1e-1), np.amax(b[particle_ind, :]) * (1 + 1e-1)])
        ax5.set_ylabel(r'$\|B\|$ (nT)')
        ax5.grid()
        
        ax6 = fig.add_subplot(gs[6, :])
        for i, j in enumerate(particle_ind):
            ax6.plot(t_v, v_pam[j, :] / sp.c, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax6.set_xlim([0, steps * dt])
        ax6.set_ylim([0, 1])
        ax6.set_ylabel(r'$v_{\parallel}/c$')
        ax6.grid()
        
        ax7 = fig.add_subplot(gs[7, :])
        for i, j in enumerate(particle_ind):
            ax7.plot(t_v, v_pem[j, :] / sp.c, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax7.set_xlim([0, steps * dt])
        ax7.set_ylim([0, 1])
        ax7.set_ylabel(r'$v_{\perp}/c$')
        ax7.grid()
        
        ax8 = fig.add_subplot(gs[8, :])
        for i, j in enumerate(particle_ind):
            ax8.plot(t_v, gr[j, :] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax8.set_xlim([0, steps * dt])
        ax8.set_ylim([0, np.amax(gr[particle_ind, :]) * inv_Re])
        ax8.set_ylabel(r'Gyroradius ($R_E$)')
        ax8.grid()
        
        ax9 = fig.add_subplot(gs[9, :])
        for i, j in enumerate(particle_ind):
            ax9.plot(t_v, gf[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax9.set_xlim([0, steps * dt])
        ax9.set_ylim([0, np.amax(gf[particle_ind, :])])
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel(r'Gyrofrequency (s$^{-1}$)')
        ax9.grid()
        
        fig.tight_layout(pad=0.4)
        handles, labels = ax9.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.show()
        
    return plot

@njit
def eq_pitch_angle_from_moment(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    mom = magnetic_moment(history, intrinsic)
    bm  = b_mag(history)
    # pa  = np.radians(pitch_angle(history))
    v   = history[:, :, 1, :]
    
    history_new = np.zeros((num_particles, steps))

    for i in range(num_particles):
        b_min = np.amin(bm[i])
        for j in range(steps):
            history_new[i, j] = np.arcsin(np.sqrt(mom[i, j] * 2 * b_min / (intrinsic[i, 0] * dot(v[i, j], v[i, j]))))
            # history_new[i, j] = np.arcsin(np.sqrt(b_min / bm[i, j]) * np.sin(pa[i, j]))
            
    return np.degrees(history_new)


@njit
def get_eq_pas(field, history, intrinsic, threshold=0.1):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])
    
    eq_pa = eq_pitch_angle_from_moment(history, intrinsic)
    r = position(history)
    K = kinetic_energy(history, intrinsic)
    pas_old = np.zeros((num_particles, steps, 3)) - 1
    
    max_crossings = 0
    
    for i in range(num_particles):
        K_max = np.amax(K[i, :]) 
        ad_param = adiabaticity(field, r[i, :, :], K_max)
        contig_args = np.argwhere(ad_param <= threshold)[:, 0]
        
        if len(contig_args) == 0:
            continue

        disc_args = np.argwhere(np.diff(contig_args) != 1)[:, 0]
                
        args = np.zeros(2 * len(disc_args) + 2)
        args[0] = contig_args[0]
        for j in range(len(disc_args)):
            args[2 * j + 1] = contig_args[disc_args[j]]
            args[2 * j + 2] = contig_args[disc_args[j] + 1]
        args[-1] = contig_args[-1]
                
        vals = np.unique(args)
        dup_args = []
        count = []
        
        for v in vals:
            a = np.argwhere(args == v)[:, 0]
            if len(a) > 1:
                dup_args.append(a[0])
                count.append(len(a))
                        
        for j in range(len(dup_args)):
            total_count = 0
            for k in range(j):
                total_count += count[k]
            args = np.delete(args, np.arange(dup_args[j] - total_count, dup_args[j] + count[j] - total_count))
                    
        if int(len(args) / 2) > max_crossings:
            max_crossings = int(len(args) / 2)
        
        for j in range(int(len(args) / 2)):
            pas_old[i, j, 0] = np.mean(eq_pa[i, int(args[2 * j]):int(args[2 * j + 1])])
            pas_old[i, j, 1] = args[2 * j]
            pas_old[i, j, 2] = args[2 * j + 1]
            
    pas_new = np.zeros((num_particles, max_crossings, 3)) - 1
    pas_new[:, :, :] = pas_old[:, 0:max_crossings, :]
                
    return pas_new

def get_pas_at_bounce_phase(history, phase):
    pas = pitch_angle(history)
    
    bounce_pas = []
    
    nth_crossing = int(phase // np.pi)
    additional_phase = phase % np.pi

    for i in range(len(pas[:, 0])):
        zero_crossings = np.where(np.diff(np.sign(pas[i] - 90)))[0]
        max_crossings = len(zero_crossings) - 1
        
        if max_crossings >= nth_crossing + 1:
            diff = zero_crossings[nth_crossing + 1] - zero_crossings[nth_crossing]
            eq_point = np.abs(pas[i, zero_crossings[nth_crossing]:zero_crossings[nth_crossing] + diff] - 90).argmax()
            first_half = eq_point
            second_half = diff - eq_point

            if additional_phase <= np.pi / 2:
                ind = int(first_half / (np.pi / 2) * additional_phase) + zero_crossings[nth_crossing]
                bounce_pas.append(pas[i, ind])
            else:
                ind = int(first_half + second_half / (np.pi / 2) * (additional_phase - np.pi / 2)) + zero_crossings[nth_crossing]
                bounce_pas.append(pas[i, ind])
    
    return bounce_pas


def get_pas_at_bounce_phase_all_t(history, phase):
    all_pas = get_pas_at_bounce_phase(history, phase)
    
    while True:
        phase = phase + 2 * np.pi
        new_pas = get_pas_at_bounce_phase(history, phase)

        if len(new_pas) == 0:
            break
        else:
            for pa in new_pas:
                all_pas.append(pa)
                
    return all_pas


def gca_filter(history, intrinsic, dt):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    position = history[:, :, 0]
    gyrofrequency_list = gyrofreq(history, intrinsic)

    history_new =  np.zeros((num_particles, steps, 3))
    
    for i in range(num_particles):
        b, a = signal.butter(4, np.amin(gyrofrequency_list[i]) / (2 * np.pi) * 0.1, fs=(1. / dt))
        zi = signal.lfilter_zi(b, a)

        x, _ = signal.lfilter(b, a, position[i, :, 0], zi=zi*position[i, 0, 0])
        y, _ = signal.lfilter(b, a, position[i, :, 1], zi=zi*position[i, 0, 1])
        z, _ = signal.lfilter(b, a, position[i, :, 2], zi=zi*position[i, 0, 2])

        history_new[i, :, 0] = x
        history_new[i, :, 1] = y
        history_new[i, :, 2] = z

    return history_new
