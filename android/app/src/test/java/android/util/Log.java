package android.util;

/**
 * Stub android.util.Log for plain JUnit tests.
 * Android classes reference Log throughout; this stub prevents
 * "ClassNotFoundException" when running tests outside Android.
 */
public final class Log {
    public static int v(String tag, String msg) { return 0; }
    public static int d(String tag, String msg) { return 0; }
    public static int i(String tag, String msg) { return 0; }
    public static int w(String tag, String msg) { return 0; }
    public static int e(String tag, String msg) { return 0; }
    public static int d(String tag, String msg, Throwable tr) { return 0; }
    public static int e(String tag, String msg, Throwable tr) { return 0; }
}
